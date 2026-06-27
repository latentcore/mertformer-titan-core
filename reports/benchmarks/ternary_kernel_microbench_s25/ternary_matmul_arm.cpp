// =============================================================================
//  MertFormer Ternary Matmul — ARM / NEON  (Galaxy S25 vb.)  [v2: oto-turbo]
//  BitNet b1.58: agirliklar {-1, 0, +1}
// -----------------------------------------------------------------------------
//  KANIT NOTU (scope): Bu, repo'nun ANA YOLU DEGILDIR. Kanonik egitim/cikarim
//  yolu bitlinear.py (PyTorch STE) + triton_fused_bitlinear.py (GPU) uzerindedir.
//  Bu dosya, S25 CPU'da tek-op ternary matmul'un OLCULMUS bir mikrobenchmark'idir
//  (full-model t/s veya NPU degil). results.json + README.md ile birlikte okunur.
// -----------------------------------------------------------------------------
//  Telefon islemcisi ARM'dir => Intel immintrin.h / AVX-512 / VNNI BURADA YOK.
//  ARM karsiliklari: NEON (128-bit) ve SDOT (vdotq = int8 nokta-carpim, VNNI'nin
//  ARM esi). Bu surum SDOT'u inline-assembly ile cagirir; boylece ekstra derleme
//  BAYRAGINA GEREK KALMADAN, CPU destekliyorsa turbo KENDILIGINDEN acilir.
//
//  KADEME 1 — NEON FMA EXACT : float'a bit-bire-bir esit (fark=0). HILE YOK.
//  KADEME 2 — SDOT TURBO     : int8 aktivasyon, 16 MAC/komut, ~%0.4 rms. ~8x.
//  Calisma aninda getauxval(HWCAP) ile dotprod var mi diye bakilir; yoksa atlanir.
//
//  ADALET: ayni veri/seed; -ffast-math YOK; yer-gercegi double.
//
//  CxxDroid'de DERLEME:  sadece  -O3  yeterli (turbo otomatik).
//     Komut satiri:  g++ -O3 ternary_matmul_arm.cpp -o tern && ./tern
//     (Istege bagli ekstra hiz:  -O3 -march=armv8.2-a+dotprod)
// =============================================================================
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
#include <chrono>
#include <cmath>

#if defined(__ARM_NEON) || defined(__aarch64__)
  #include <arm_neon.h>
  #define HAVE_NEON 1
  #include <sys/auxv.h>
  #ifndef HWCAP_ASIMDDP
  #define HWCAP_ASIMDDP (1u<<20)
  #endif
  #if defined(__clang__)
    #define DOTPROD_FN __attribute__((target("dotprod")))
  #else
    #define DOTPROD_FN __attribute__((target("+dotprod")))
  #endif
#endif

using namespace std;
using namespace std::chrono;

// MR -> N'i, NR -> NBn'yi BOLMELI (yoksa bellek tasmasi). N=256 icin MR in {4,8}.
static const int N=256, ITERS=8;
static const int NBn=N/4, KG=N/4;

alignas(64) static float   x[N*N], wf[N*N], out[N*N], reff[N*N];
alignas(64) static int8_t  wq[N*N], xq[N*N], bpk[NBn*KG*16];
alignas(64) static double  refd[N*N];

static inline double now_ms(){return duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count()/1e6;}
template<class F> double bench(F f,int reps=15){f();f();double best=1e30;for(int r=0;r<reps;r++){double t0=now_ms();f();double t=now_ms()-t0;if(t<best)best=t;}return best;}
double errF(){double m=0;for(int i=0;i<N*N;i++){double d=fabs((double)out[i]-(double)reff[i]);if(d>m)m=d;}return m;}
double errD(){double m=0;for(int i=0;i<N*N;i++){double d=fabs((double)out[i]-refd[i]);if(d>m)m=d;}return m;}
double rmsD(){double s=0;for(int i=0;i<N*N;i++){double d=(double)out[i]-refd[i];s+=d*d;}return sqrt(s/(N*N));}

// [0] float-naive referans (k sirali)
void f_naive(){
  for(int it=0;it<ITERS;it++)for(int i=0;i<N;i++)for(int j=0;j<N;j++){
    float a=0; for(int k=0;k<N;k++) a+=x[i*N+k]*wf[k*N+j]; out[i*N+j]=a;}
}

#ifdef HAVE_NEON
// KADEME 1 — NEON FMA EXACT (4 cikti sutunu/lane, k SIRALI => bit-exact)
template<int MR,int NR>
void neon_fma(){
  for(int it=0;it<ITERS;it++)
   for(int jb=0;jb<NBn;jb+=NR)
    for(int i=0;i<N;i+=MR){
      float32x4_t acc[MR][NR];
      for(int a=0;a<MR;a++)for(int b=0;b<NR;b++)acc[a][b]=vdupq_n_f32(0);
      for(int k=0;k<N;k++){
        float32x4_t wv[NR];
        for(int b=0;b<NR;b++)wv[b]=vld1q_f32(&wf[k*N+(jb+b)*4]);
        for(int a=0;a<MR;a++){float32x4_t xb=vdupq_n_f32(x[(i+a)*N+k]);
          for(int b=0;b<NR;b++)acc[a][b]=vfmaq_f32(acc[a][b],xb,wv[b]);}
      }
      for(int a=0;a<MR;a++)for(int b=0;b<NR;b++)vst1q_f32(&out[(i+a)*N+(jb+b)*4],acc[a][b]);
    }
}

// SDOT (inline-asm): acc.4s += sum_4( a.16b * b.16b ) lane-grup. Bayraksiz calisir.
DOTPROD_FN static inline int32x4_t sdot4(int32x4_t acc,int8x16_t a,int8x16_t b){
  __asm__("sdot %0.4s, %1.16b, %2.16b":"+w"(acc):"w"(a),"w"(b));
  return acc;
}
// KADEME 2 — SDOT TURBO
template<int MR,int NR>
DOTPROD_FN void neon_dot(float sx){
  for(int it=0;it<ITERS;it++)
   for(int i=0;i<N;i+=MR)
    for(int jb=0;jb<NBn;jb+=NR){
      int32x4_t acc[MR][NR];
      for(int a=0;a<MR;a++)for(int b=0;b<NR;b++)acc[a][b]=vdupq_n_s32(0);
      for(int g=0;g<KG;g++){
        int8x16_t bv[NR];
        for(int b=0;b<NR;b++)bv[b]=vld1q_s8(&bpk[((jb+b)*KG+g)*16]);
        for(int a=0;a<MR;a++){
          int32_t four=*(const int32_t*)&xq[(i+a)*N+g*4];
          int8x16_t av=vreinterpretq_s8_s32(vdupq_n_s32(four));
          for(int b=0;b<NR;b++)acc[a][b]=sdot4(acc[a][b],av,bv[b]);
        }
      }
      for(int a=0;a<MR;a++)for(int b=0;b<NR;b++)
        vst1q_f32(&out[(i+a)*N+(jb+b)*4], vmulq_n_f32(vcvtq_f32_s32(acc[a][b]),sx));
    }
}
#endif

// int8 nicemleme (signed; sdot icin offset gerekmez)
float quantize(){
  float mx=0; for(int i=0;i<N*N;i++){float a=fabsf(x[i]); if(a>mx)mx=a;}
  float sx=(mx>0)?mx/127.f:1.f, inv=(mx>0)?127.f/mx:0.f;
  for(int i=0;i<N*N;i++){int q=(int)lrintf(x[i]*inv); if(q>127)q=127; if(q<-127)q=-127; xq[i]=(int8_t)q;}
  return sx;
}

int main(){
  setvbuf(stdout,NULL,_IONBF,0);
  mt19937 rng(1453); uniform_real_distribution<float>fd(-1,1); uniform_int_distribution<int>td(-1,1);
  for(int i=0;i<N*N;i++){x[i]=fd(rng);int t=td(rng);wq[i]=(int8_t)t;wf[i]=(float)t;}
  // SDOT agirlik paketi: bpk[(jb*KG+g)*16 + l*4 + t] = w[g*4+t][jb*4+l]
  for(int jb=0;jb<NBn;jb++)for(int g=0;g<KG;g++)for(int l=0;l<4;l++)for(int t=0;t<4;t++)
    bpk[(jb*KG+g)*16+l*4+t]=wq[(g*4+t)*N+jb*4+l];
  for(int i=0;i<N;i++)for(int j=0;j<N;j++){double a=0;for(int k=0;k<N;k++)a+=(double)x[i*N+k]*(double)wf[k*N+j];refd[i*N+j]=a;}

  printf("===========================================================\n");
  printf("  MertFormer Ternary Matmul - ARM/NEON  (N=%d ITERS=%d)\n",N,ITERS);
  printf("  Adalet: ayni veri/seed; -ffast-math YOK; yer-gercegi=double\n");
  printf("===========================================================\n");

  {volatile double s=0;for(int i=0;i<3;i++){f_naive();s+=out[0];}}
  f_naive(); memcpy(reff,out,sizeof(out));
  double tF=bench(f_naive);
  printf("[0] float-naive (REFERANS)  %9.3f ms   1.00x\n\n",tF);

#ifdef HAVE_NEON
  printf("--- KADEME 1 : NEON FMA EXACT (bit-bire-bir float; HILE YOK) ---\n");
  {double t=bench(neon_fma<4,4>);printf("  NEON FMA 4x4  %9.3f ms  %5.2fx  float-fark=%.2e  double=%.2e\n",t,tF/t,errF(),errD());}
  {double t=bench(neon_fma<8,2>);printf("  NEON FMA 8x2  %9.3f ms  %5.2fx  float-fark=%.2e\n",t,tF/t,errF());}
  {double t=bench(neon_fma<4,8>);printf("  NEON FMA 4x8  %9.3f ms  %5.2fx  float-fark=%.2e\n",t,tF/t,errF());}

  bool has_dp=(getauxval(AT_HWCAP)&HWCAP_ASIMDDP)!=0;
  printf("\n--- KADEME 2 : SDOT TURBO (int8, ~%%0.4 rms)  [CPU dotprod: %s] ---\n", has_dp?"VAR":"YOK");
  if(has_dp){
    float sx=quantize();
    {double t=bench([&]{neon_dot<4,4>(sx);});printf("  SDOT 4x4  %9.3f ms  %5.2fx  rms=%.2e\n",t,tF/t,rmsD());}
    {double t=bench([&]{neon_dot<8,2>(sx);});printf("  SDOT 8x2  %9.3f ms  %5.2fx\n",t,tF/t);}
    {double t=bench([&]{neon_dot<4,8>(sx);});printf("  SDOT 4x8  %9.3f ms  %5.2fx\n",t,tF/t);}
  } else {
    printf("  (Bu CPU dotprod desteklemiyor - turbo atlandi)\n");
  }
#else
  printf("--- NEON yok (bu platformda) ---\n");
#endif

  printf("\n===========================================================\n");
  printf("  FMA EXACT = float ile MATEMATIKSEL OZDES (fark=0), SIMD ile\n");
  printf("  hizli. SDOT = int8 ternary compute tavani. Hile yok.\n");
  printf("===========================================================\n");
  return 0;
}
