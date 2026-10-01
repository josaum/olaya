---
name: building-nice-landing-pages
description: >-
  Expert guidelines and design system standards for crafting world-class, high-converting,
  ultra-polished landing pages for developer tools, AI infrastructure, and software products.
  Combines the Julian Shapiro Conversion Framework, Linear/Stripe design aesthetics,
  high-density bento grids, micro-interactions, and technical credibility proofs.
---

# Building World-Class Developer & AI Landing Pages

This skill outlines the exact aesthetic standards, information architecture, and conversion mechanics required to design and build world-class landing pages for technical tools, AI systems, and developer infrastructure.

Inspired by industry benchmarks: **Linear**, **Stripe**, **Modal**, **Vercel**, **Scale AI**, and the **Julian Shapiro Conversion Framework**.

---

## 1. The Fundamental Law of Conversion

$$\text{Conversion Rate} = \text{Desire} - (\text{Labor} + \text{Confusion})$$

1. **Maximize Desire**:
   - Deliver visceral "magic moments" above the fold within 3 seconds.
   - Show, don't tell: live in-browser simulations, interactive widgets, benchmark head-to-heads.
   - Ground claims in formal guarantees (e.g. mathematical proofs, exact benchmark numbers, open weights).

2. **Minimize Labor**:
   - Zero-barrier onboarding: 1-click interactive demo (no login, no API key required).
   - Universal 1-click copy buttons on all code blocks, commands, and cURL requests.
   - Single-fold access to primary actions.

3. **Minimize Confusion**:
   - Crystal-clear positioning: "What it is" and "What it is NOT" (e.g., *Leio is not a chatbot; it returns mathematically proven decisions*).
   - Boring consistency: strict typography scale, consistent padding, uniform border-radius.
   - Direct objection handling: latency, scale, cost, deployment options.

---

## 2. Design System & Aesthetic Principles

### The Engineering Dark Aesthetic (Linear Style)
- **Palette**:
  - Canvas background: `#0d1117` (GitHub/Linear deep charcoal)
  - Card/Panel background: `#161b22` with subtle transparency (`rgba(22, 27, 34, 0.8)`)
  - Border tokens:
    - Default border: `1px solid #30363d`
    - Muted border: `1px solid #21262d`
    - High-tech glowing active border: `1px solid rgba(56, 139, 253, 0.5)`
  - Accent Colors:
    - Primary electric blue: `#58a6ff`
    - Success/verification emerald: `#3fb950`
    - Category purple: `#bc8cff`
    - Warning/caution amber: `#f0883e`
    - Critical red: `#f85149`

### Typography Hierarchy (Inter / System Font Stack)
- `font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;`
- Strict typographic scale:
  - Hero H1: `2.2rem - 2.5rem`, `letter-spacing: -0.035em`, `line-height: 1.15`
  - Section H2: `1.4rem - 1.55rem`, `letter-spacing: -0.02em`, `border-bottom: 1px solid #21262d`
  - Card/Feature H3: `1.05rem - 1.15rem`, `color: #e6edf3`
  - Body copy: `15px / 1.65`, `color: #c9d1d9`
  - Monospace telemetry / code: `ui-monospace, SFMono-Regular, Menlo, Monaco, monospace`

### Micro-Interactions & Visual Polish
- **Ambient Lighting**: Subtle radial gradient glows behind headers (`radial-gradient(...)`).
- **Pulsing Status Dots**: Live animated indicator dots representing running state.
- **Glassmorphism**: Backdrop blur (`backdrop-filter: blur(10px)`) on fixed navbars and floating badges.
- **Interactive States**: Hover transitions with slight elevation (`transform: translateY(-1px)`), border luminescence.

---

## 3. Seven-Fold Landing Page Architecture

1. **Fold 1: The Command Center (Fixed Navbar & Hero)**:
   - Fixed 48px glassmorphic navbar with brand pill, live status indicator, anchor links, and glowing CTA.
   - High-contrast Hero: Headline + Sub-headline + Stat pills + Dual action buttons + Head-to-Head Duel.
2. **Fold 2: The Physical / Interactive Demonstration**:
   - A live functional simulation or visual workbench (e.g. warehouse conveyor sorter with laser scanner).
3. **Fold 3: The Interactive Dual-View Decision Workbench**:
   - Hands-on workbench allowing instant query execution with live probability bars and axiom proofs.
4. **Fold 4: Real-World Use Cases & Sub-Lattices**:
   - Concrete industry walkthroughs with domain switching (GS1 supply chain, DOID clinical, SRE triage).
5. **Fold 5: Verifiable Hardware Benchmarks**:
   - Detailed performance table with architecture, parameters, median latency, P95, and QPS throughput.
6. **Fold 6: The Zero-Install Agent Wedge & Developer Onboarding**:
   - Instant CLI commands (`uvx`, `npx`, `pip`) with 1-click copy buttons and client auto-config.
7. **Fold 7: Transparent Economics & Interactive ROI Calculator**:
   - Explicit 3-tier pricing cards and a dynamic volume slider calculating annual dollar & latency savings.

---

## 4. Quality Checklist

- [ ] Does the page answer "What is this and why should I care?" in < 5 seconds?
- [ ] Is there an interactive element visible above or immediately at the fold?
- [ ] Are all code snippets 1-click copyable with visual checkmark feedback?
- [ ] Is mobile responsiveness flawless (no page blowout, horizontal scrolling, or clipped text)?
- [ ] Are all performance metrics backed by concrete hardware and batch size specifications?
- [ ] Is the primary CTA unmissable with consistent destination?
