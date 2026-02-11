# White Paper: LiquidRouter Architecture
**Temporal-Aware Routing for Sparse Mixture of Experts**

## 1. Abstract
The MertFormer Titan architecture introduces the **LiquidRouter**, a novel gating mechanism for Sparse Mixture of Experts (MoE) that leverages Closed-Form Continuous-time (CfC) neural networks. Unlike traditional "stateless" routers that treat tokens as independent events, LiquidRouter preserves temporal momentum, significantly improving expert utilization and routing stability in on-device AI applications.

## 2. The Problem: Stateless MoE Collapse
Traditional MoE routers use a simple linear projection followed by a Softmax to select experts. This approach suffers from:
- **Expert Collapse**: Over-utilization of a few experts due to lack of historical context.
- **Inference Jitter**: Rapidly switching experts between tokens leads to cache misses and increased latency on NPU hardware.

## 3. The Solution: CfC-Based Liquid Routing
LiquidRouter replaces the standard gating network with a **Liquid Neural Network (LNN)** cell. By modeling the routing decision as a continuous differential equation, the system gains:
- **Temporal Context**: The choice of an expert for token $x_t$ is influenced by the hidden state and momentum of tokens $x_{t-1 \dots t-n}$.
- **Smooth Transitions**: The "fluid" nature of CfC ensures that routing decisions evolve logically, reducing hardware-level context switching.

### Mathematical Foundation
The routing weight $G(t)$ is calculated as:
$$G(t) = \sigma(CfC(x_t, h_{t-1}))$$
Where $CfC$ represents the Closed-Form solution to the neural ODE, allowing for efficient, hardware-aware computation that tracks the "flow" of data.

## 4. Hardware Sinergy
On NPUs (like the Snapdragon 8 Elite), LiquidRouter optimizes energy consumption by:
1. **Predictive Activation**: Pre-calculating likely expert paths before the token fully arrives.
2. **Reduced Switching**: Minimizing the high-energy cost of loading new expert weights into the NPU's local memory.

## 5. Conclusion
LiquidRouter is a strategic moat for MertFormer Titan, providing the first empirically stable, on-device MoE architecture that respects the temporal nature of language.
