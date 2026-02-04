"""
Build minimal investor decks (EN/TR) as PPTX without external dependencies.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from xml.sax.saxutils import escape

OUTPUT_EN = Path("reports/investor_deck.pptx")
OUTPUT_TR = Path("reports/investor_deck_TR.pptx")

SLIDE_WIDTH = 9144000
SLIDE_HEIGHT = 5143500

THEME = {
    "bg": "0F172A",   # slate-900
    "title": "7DD3FC", # sky-300
    "body": "E2E8F0",  # slate-200
    "accent": "22D3EE", # cyan-400
    "footer": "94A3B8", # slate-400
}

SLIDES_EN = [
    {
        "title": "MertFormer Titan",
        "bullets": [
            "Edge-native coding model (2.6B target)",
            "Operator-mode verified pipeline",
            "Offline-first, mobile compute focus",
        ],
    },
    {
        "title": "Problem",
        "bullets": [
            "Cloud AI is costly and slow for regulated workflows",
            "Privacy and data sovereignty are hard to guarantee",
            "Latency blocks real-time, on-device coding",
        ],
    },
    {
        "title": "Solution",
        "bullets": [
            "On-device inference with BitNet 1.58-bit quantization",
            "LiquidRouter MoE for stable, efficient routing",
            "Long-context attention optimized for code",
        ],
    },
    {
        "title": "Product",
        "bullets": [
            "2.6B parameter target model for coding",
            "Offline operation on mobile-class hardware",
            "Security-first with reproducibility gates",
        ],
    },
    {
        "title": "Architecture Highlights",
        "bullets": [
            "BitLinear + Liquid Neural Networks",
            "MLA attention with long-context readiness",
            "Sparse MoE with temporal routing",
        ],
    },
    {
        "title": "Safety & Reliability",
        "bullets": [
            "Kill-switch for non-finite stability events",
            "Failure budget and pivot triggers",
            "Checkpoint restore drills and reproducibility stamp",
        ],
    },
    {
        "title": "Evaluation Plan",
        "bullets": [
            "Golden sample suite (50 prompts)",
            "HumanEval/MBPP output generation",
            "Full 1MB overfit gate on training hardware",
        ],
    },
    {
        "title": "Compute Need & Plan",
        "bullets": [
            "Multi-GPU credits for master run (A100/H100 class)",
            "Phased run: full 1MB gate → master run → benchmarks",
            "Deliverables: checkpoints, logs, eval outputs",
        ],
    },
    {
        "title": "Pilot / Go-to-Market",
        "bullets": [
            "Offline PoC for regulated teams (defense/legal/finance)",
            "2–4 week pilot with clear success criteria",
            "Post‑pilot: paid deployments + support",
        ],
    },
    {
        "title": "Market & Use Cases",
        "bullets": [
            "Regulated enterprises and defense workflows",
            "On-device copilots for low-connectivity regions",
            "Private inference stacks with data control",
        ],
    },
    {
        "title": "Roadmap",
        "bullets": [
            "Master run (2.6B) with telemetry gates",
            "Benchmark validation + pilot deployments",
            "Asset stack + demo video for launch",
        ],
    },
    {
        "title": "Ask",
        "bullets": [
            "Pilot partners for edge coding workflows",
            "Infrastructure credits for training runs",
            "Strategic advisors for deployment validation",
        ],
    },
]

SLIDES_TR = [
    {
        "title": "MertFormer Titan",
        "bullets": [
            "Edge-native kodlama modeli (2.6B hedef)",
            "Operator-mode doğrulanmış pipeline",
            "Offline-öncelikli, mobil compute odaklı",
        ],
    },
    {
        "title": "Problem",
        "bullets": [
            "Bulut AI düzenlemeli işlerde pahalı ve yavaş",
            "Gizlilik ve veri egemenliği zor garanti edilir",
            "Gecikme, gerçek zamanlı cihaz‑içi kodlamayı engeller",
        ],
    },
    {
        "title": "Çözüm",
        "bullets": [
            "BitNet 1.58‑bit kuantizasyon ile cihaz‑içi çıkarım",
            "LiquidRouter MoE ile stabil ve verimli yönlendirme",
            "Kod için optimize uzun bağlam dikkat",
        ],
    },
    {
        "title": "Ürün",
        "bullets": [
            "Kodlama için 2.6B parametre hedef model",
            "Mobil sınıf donanımda offline çalışma",
            "Reproducibility gate’leriyle güvenlik‑öncelik",
        ],
    },
    {
        "title": "Mimari Öne Çıkanlar",
        "bullets": [
            "BitLinear + Liquid Neural Networks",
            "MLA dikkat ile uzun bağlam hazırlığı",
            "Zaman‑bağımlı yönlendirmeli seyrek MoE",
        ],
    },
    {
        "title": "Güvenlik & Dayanıklılık",
        "bullets": [
            "Non‑finite olaylar için kill‑switch",
            "Failure budget ve pivot tetikleyiciler",
            "Checkpoint restore drill ve reproducibility stamp",
        ],
    },
    {
        "title": "Değerlendirme Planı",
        "bullets": [
            "Golden sample seti (50 prompt)",
            "HumanEval/MBPP çıktı üretimi",
            "Eğitim donanımında 1MB overfit gate",
        ],
    },
    {
        "title": "Compute İhtiyacı & Plan",
        "bullets": [
            "Master run için multi-GPU kredisi (A100/H100 sınıfı)",
            "Aşamalı koşu: full 1MB gate → master run → benchmark",
            "Çıktılar: checkpoint, log, eval sonuçları",
        ],
    },
    {
        "title": "Pilot / Pazara Çıkış",
        "bullets": [
            "Düzenlemeli ekipler için offline PoC (savunma/hukuk/finans)",
            "2–4 haftalık pilot, net başarı kriterleriyle",
            "Pilot sonrası: ücretli dağıtım + destek",
        ],
    },
    {
        "title": "Pazar & Kullanım",
        "bullets": [
            "Düzenlemeli kurumlar ve savunma iş akışları",
            "Düşük bağlantı bölgelerinde cihaz‑içi yardımcılar",
            "Veri kontrolü olan özel inference stack’leri",
        ],
    },
    {
        "title": "Yol Haritası",
        "bullets": [
            "Telemetri gate’leri ile 2.6B master run",
            "Benchmark doğrulama + pilot dağıtımlar",
            "Lansman için asset stack + demo video",
        ],
    },
    {
        "title": "Talep",
        "bullets": [
            "Edge coding pilot ortakları",
            "Eğitim koşuları için altyapı kredisi",
            "Dağıtım doğrulaması için stratejik danışmanlar",
        ],
    },
]

NS_P = "http://schemas.openxmlformats.org/presentationml/2006/main"
NS_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
NS_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def paragraph(text: str, size: int, color: str, bold: bool = False, bullet: bool = False) -> str:
    bold_attr = ' b="1"' if bold else ''
    return (
        f"<a:p><a:pPr>{'<a:buChar char="•"/>' if bullet else '<a:buNone/>'}</a:pPr>"
        f"<a:r><a:rPr sz=\"{size}\"{bold_attr}><a:solidFill><a:srgbClr val=\"{color}\"/></a:solidFill></a:rPr>"
        f"<a:t>{escape(text)}</a:t></a:r><a:endParaRPr sz=\"{size}\"/>"
        f"</a:p>"
    )


def text_box(shape_id: int, name: str, x: int, y: int, cx: int, cy: int, paragraphs: str) -> str:
    return (
        f"<p:sp>"
        f"<p:nvSpPr><p:cNvPr id=\"{shape_id}\" name=\"{escape(name)}\"/>"
        f"<p:cNvSpPr txBox=\"1\"/><p:nvPr/></p:nvSpPr>"
        f"<p:spPr><a:xfrm><a:off x=\"{x}\" y=\"{y}\"/>"
        f"<a:ext cx=\"{cx}\" cy=\"{cy}\"/></a:xfrm>"
        f"<a:prstGeom prst=\"rect\"><a:avLst/></a:prstGeom></p:spPr>"
        f"<p:txBody><a:bodyPr wrap=\"square\"/><a:lstStyle/>"
        f"{paragraphs}"
        f"</p:txBody>"
        f"</p:sp>"
    )


def solid_rect(shape_id: int, name: str, x: int, y: int, cx: int, cy: int, color: str) -> str:
    return (
        f"<p:sp>"
        f"<p:nvSpPr><p:cNvPr id=\"{shape_id}\" name=\"{escape(name)}\"/>"
        f"<p:cNvSpPr/><p:nvPr/></p:nvSpPr>"
        f"<p:spPr><a:xfrm><a:off x=\"{x}\" y=\"{y}\"/>"
        f"<a:ext cx=\"{cx}\" cy=\"{cy}\"/></a:xfrm>"
        f"<a:prstGeom prst=\"rect\"><a:avLst/></a:prstGeom>"
        f"<a:solidFill><a:srgbClr val=\"{color}\"/></a:solidFill>"
        f"<a:ln><a:noFill/></a:ln>"
        f"</p:spPr><p:txBody><a:bodyPr/><a:lstStyle/><a:p/></p:txBody>"
        f"</p:sp>"
    )


def slide_xml(title: str, bullets: list[str], footer: str) -> str:
    title_paragraphs = paragraph(title, size=3600, color=THEME["title"], bold=True, bullet=False)
    body_paragraphs = "".join(paragraph(b, size=2400, color=THEME["body"], bullet=True) for b in bullets)
    footer_paragraphs = paragraph(footer, size=1400, color=THEME["footer"], bullet=False)

    bg = solid_rect(2, "Background", 0, 0, SLIDE_WIDTH, SLIDE_HEIGHT, THEME["bg"])
    accent = solid_rect(3, "Accent", 0, 0, SLIDE_WIDTH, 120000, THEME["accent"])
    title_box = text_box(4, "Title", 457200, 228600, 8229600, 685800, title_paragraphs)
    body_box = text_box(5, "Body", 685800, 1200000, 7772400, 3200000, body_paragraphs)
    footer_box = text_box(6, "Footer", 457200, 4700000, 8229600, 300000, footer_paragraphs)

    return (
        f"<p:sld xmlns:p=\"{NS_P}\" xmlns:a=\"{NS_A}\" xmlns:r=\"{NS_R}\">"
        f"<p:cSld><p:spTree>"
        f"<p:nvGrpSpPr><p:cNvPr id=\"1\" name=\"\"/>"
        f"<p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>"
        f"<p:grpSpPr><a:xfrm><a:off x=\"0\" y=\"0\"/>"
        f"<a:ext cx=\"0\" cy=\"0\"/></a:xfrm></p:grpSpPr>"
        f"{bg}{accent}{title_box}{body_box}{footer_box}"
        f"</p:spTree></p:cSld>"
        f"<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>"
        f"</p:sld>"
    )


def write_pptx(output_path: Path, slides: list[dict[str, list[str]]], footer: str) -> None:
    # Build a self-contained PPTX package (XML + relationships) without external dependencies.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    content_types = [
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>",
        "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">",
        "<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>",
        "<Default Extension=\"xml\" ContentType=\"application/xml\"/>",
        "<Override PartName=\"/ppt/presentation.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml\"/>",
        "<Override PartName=\"/ppt/slideMasters/slideMaster1.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml\"/>",
        "<Override PartName=\"/ppt/slideLayouts/slideLayout1.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml\"/>",
        "<Override PartName=\"/ppt/theme/theme1.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.theme+xml\"/>",
        "<Override PartName=\"/docProps/core.xml\" ContentType=\"application/vnd.openxmlformats-package.core-properties+xml\"/>",
        "<Override PartName=\"/docProps/app.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.extended-properties+xml\"/>",
    ]
    for idx in range(1, len(slides) + 1):
        content_types.append(
            f"<Override PartName=\"/ppt/slides/slide{idx}.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.presentationml.slide+xml\"/>"
        )
    content_types.append("</Types>")

    root_rels = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
"""

    rels = [
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>",
        f"<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">",
        f"<Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster\" Target=\"slideMasters/slideMaster1.xml\"/>",
    ]

    slide_ids = []
    for idx in range(1, len(slides) + 1):
        rels.append(
            f"<Relationship Id=\"rId{idx+1}\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide\" Target=\"slides/slide{idx}.xml\"/>"
        )
        slide_ids.append((256 + idx - 1, f"rId{idx+1}"))
    rels.append("</Relationships>")

    presentation = [
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>",
        f"<p:presentation xmlns:p=\"{NS_P}\" xmlns:a=\"{NS_A}\" xmlns:r=\"{NS_R}\">",
        "<p:sldMasterIdLst>",
        "<p:sldMasterId id=\"2147483648\" r:id=\"rId1\"/>",
        "</p:sldMasterIdLst>",
        "<p:sldIdLst>",
    ]
    for slide_id, rel_id in slide_ids:
        presentation.append(f"<p:sldId id=\"{slide_id}\" r:id=\"{rel_id}\"/>")
    presentation.extend([
        "</p:sldIdLst>",
        f"<p:sldSz cx=\"{SLIDE_WIDTH}\" cy=\"{SLIDE_HEIGHT}\" type=\"screen16x9\"/>",
        "<p:notesSz cx=\"6858000\" cy=\"9144000\"/>",
        "</p:presentation>",
    ])

    slide_master = f"""<?xml version="1.0" encoding="UTF-8"?>
<p:sldMaster xmlns:p="{NS_P}" xmlns:a="{NS_A}" xmlns:r="{NS_R}">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr>
        <p:cNvPr id="1" name=""/>
        <p:cNvGrpSpPr/>
        <p:nvPr/>
      </p:nvGrpSpPr>
      <p:grpSpPr>
        <a:xfrm>
          <a:off x="0" y="0"/>
          <a:ext cx="0" cy="0"/>
        </a:xfrm>
      </p:grpSpPr>
    </p:spTree>
  </p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
  <p:sldLayoutIdLst>
    <p:sldLayoutId id="2147483649" r:id="rId1"/>
  </p:sldLayoutIdLst>
</p:sldMaster>
"""

    slide_master_rels = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>
"""

    slide_layout = f"""<?xml version="1.0" encoding="UTF-8"?>
<p:sldLayout xmlns:p="{NS_P}" xmlns:a="{NS_A}" xmlns:r="{NS_R}" type="title" preserve="1">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr>
        <p:cNvPr id="1" name=""/>
        <p:cNvGrpSpPr/>
        <p:nvPr/>
      </p:nvGrpSpPr>
      <p:grpSpPr>
        <a:xfrm>
          <a:off x="0" y="0"/>
          <a:ext cx="0" cy="0"/>
        </a:xfrm>
      </p:grpSpPr>
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr>
    <a:masterClrMapping/>
  </p:clrMapOvr>
</p:sldLayout>
"""

    slide_layout_rels = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>
"""

    theme = f"""<?xml version="1.0" encoding="UTF-8"?>
<a:theme xmlns:a="{NS_A}" name="MertFormer Theme">
  <a:themeElements>
    <a:clrScheme name="MertFormer">
      <a:dk1><a:srgbClr val="000000"/></a:dk1>
      <a:lt1><a:srgbClr val="FFFFFF"/></a:lt1>
      <a:dk2><a:srgbClr val="1E293B"/></a:dk2>
      <a:lt2><a:srgbClr val="E2E8F0"/></a:lt2>
      <a:accent1><a:srgbClr val="{THEME['accent']}"/></a:accent1>
      <a:accent2><a:srgbClr val="7DD3FC"/></a:accent2>
      <a:accent3><a:srgbClr val="94A3B8"/></a:accent3>
      <a:accent4><a:srgbClr val="38BDF8"/></a:accent4>
      <a:accent5><a:srgbClr val="0EA5E9"/></a:accent5>
      <a:accent6><a:srgbClr val="22C55E"/></a:accent6>
      <a:hlink><a:srgbClr val="0EA5E9"/></a:hlink>
      <a:folHlink><a:srgbClr val="64748B"/></a:folHlink>
    </a:clrScheme>
    <a:fontScheme name="Office">
      <a:majorFont><a:latin typeface="Calibri"/></a:majorFont>
      <a:minorFont><a:latin typeface="Calibri"/></a:minorFont>
    </a:fontScheme>
    <a:fmtScheme name="Office"/>
  </a:themeElements>
</a:theme>
"""

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    core = f"""<?xml version="1.0" encoding="UTF-8"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>MertFormer Titan Investor Deck</dc:title>
  <dc:creator>MertFormer</dc:creator>
  <cp:lastModifiedBy>MertFormer</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>
"""

    app = """<?xml version="1.0" encoding="UTF-8"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>MertFormer Deck Builder</Application>
</Properties>
"""

    with ZipFile(output_path, "w", ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", "\n".join(content_types))
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("ppt/presentation.xml", "\n".join(presentation))
        zf.writestr("ppt/_rels/presentation.xml.rels", "\n".join(rels))
        zf.writestr("ppt/slideMasters/slideMaster1.xml", slide_master)
        zf.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", slide_master_rels)
        zf.writestr("ppt/slideLayouts/slideLayout1.xml", slide_layout)
        zf.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", slide_layout_rels)
        zf.writestr("ppt/theme/theme1.xml", theme)
        zf.writestr("docProps/core.xml", core)
        zf.writestr("docProps/app.xml", app)

        for idx, slide in enumerate(slides, 1):
            zf.writestr(f"ppt/slides/slide{idx}.xml", slide_xml(slide["title"], slide["bullets"], footer))
            zf.writestr(
                f"ppt/slides/_rels/slide{idx}.xml.rels",
                """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">
  <Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout\" Target=\"../slideLayouts/slideLayout1.xml\"/>
</Relationships>
""",
            )


def main() -> None:
    write_pptx(OUTPUT_EN, SLIDES_EN, footer="Confidential — MertFormer Titan")
    write_pptx(OUTPUT_TR, SLIDES_TR, footer="Gizli — MertFormer Titan")
    print(f"Deck written to {OUTPUT_EN} and {OUTPUT_TR}")


if __name__ == "__main__":
    main()
