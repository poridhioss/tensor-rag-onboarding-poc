# TensorRAG Onboarding — Full Implementation Plan

> This document covers the complete onboarding feature: documentation content, onboarding dialog flow, guide modal, and frontend integration into the existing tensorrag codebase.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Onboarding Dialog Flow](#2-onboarding-dialog-flow)
3. [Getting Started Guide (Documentation)](#3-getting-started-guide-documentation)
4. [Frontend File Structure](#4-frontend-file-structure)
5. [Component Architecture](#5-component-architecture)
6. [Data Files Specification](#6-data-files-specification)
7. [OnboardingDialog Component Spec](#7-onboardingdialog-component-spec)
8. [GuideModal Component Spec](#8-guidemodal-component-spec)
9. [Header Modification](#9-header-modification)
10. [Page Integration](#10-page-integration)
11. [State Management](#11-state-management)
12. [Media Assets (Future)](#12-media-assets-future)
13. [Implementation Order](#13-implementation-order)
14. [Verification Checklist](#14-verification-checklist)

---

## 1. Overview

### Problem
New users landing on `/tensorrag` have zero guidance. They see an empty canvas and don't know:
- What TensorRAG is
- How to create a project
- How to write card code
- How to build and run a pipeline

### Solution
Two-layer onboarding experience:

1. **Onboarding Dialog** — A 5-step stepper overlay that auto-shows on first visit. Gives a quick visual tour of the main concepts (write cards, build pipeline, configure, run, view results).

2. **Getting Started Guide** — A full-screen documentation modal accessible via "Learn more" link in the onboarding dialog OR via a BookOpen button in the header. Walks through building the `neural-net-pipeline` demo project step-by-step with all 9 card code snippets.

### Key Decisions
- **No Zustand store** — All onboarding state lives in `page.tsx` via `useState`. It's simple enough.
- **localStorage persistence** — Key `tensorrag-onboarding-seen` tracks whether user has seen the onboarding. Dialog auto-shows only on first visit.
- **Re-openable** — BookOpen button in header lets users re-trigger the onboarding anytime.
- **Portal rendering** — Both modals use `createPortal(…, document.body)` to escape layout constraints.
- **Follows existing patterns** — PremiumLock.tsx (overlay style), OutputModal.tsx (portal + escape key).

---

## 2. Onboarding Dialog Flow

### Step-by-step content (5 steps):

| Step | ID | Title | Headline Badge | Description |
|------|----|-------|---------------|-------------|
| 1 | `welcome` | Welcome to TensorRAG | Visual ML Pipeline Editor | TensorRAG lets you build, configure, and run machine learning pipelines visually. Write Python card code, connect them on a canvas, and execute end-to-end workflows — all from your browser. |
| 2 | `write-cards` | Write Cards | Editor View | Switch to the Editor view to create projects, organize cards into folders, and write Python code. Each card is a self-contained processing step with typed inputs, outputs, and configurable parameters. |
| 3 | `build-pipeline` | Build Pipeline | Board View | Switch to the Board view to visually assemble your pipeline. Drag cards from the palette onto the canvas, then connect output handles to input handles to define data flow between steps. |
| 4 | `configure-run` | Configure & Run | Set Parameters | Click any card on the canvas to open its configuration panel. Set parameters like dataset URLs, model architectures, or training epochs. When ready, hit Run All to execute the entire pipeline. |
| 5 | `view-results` | View Results | Real-time Feedback | Monitor execution progress in real time. Each card shows its status — pending, running, completed, or failed. Click a completed card to preview its output: data tables, model summaries, metrics, and more. |

### User interactions:
- **Next** button advances to next step
- **Previous** button goes back
- **Step dots** at top show progress (active = purple, inactive = zinc)
- **"Learn more"** link at bottom opens the Guide Modal on top
- **"Get Started"** button on last step dismisses the dialog
- **X** button (top-right) closes at any time
- **Escape** key closes

### First-visit behavior:
1. Page loads → `useEffect` checks `localStorage.getItem("tensorrag-onboarding-seen")`
2. If not set → `setShowOnboarding(true)`
3. When dismissed (X, Escape, or "Get Started") → `localStorage.setItem("tensorrag-onboarding-seen", "true")` + `setShowOnboarding(false)`

---

## 3. Getting Started Guide (Documentation)

The guide is based entirely on `NEURAL_NETWORK_PIPELINE.md`. It's rendered as a React component (not MDX) so it can live inside the existing component tree without additional tooling.

### 6 Sections:

#### Section 1: Creating a Project
- How to switch to Editor view
- How to create a new project named `neural-net-pipeline`
- How to create the 5 folders: `data/`, `model/`, `training/`, `evaluation/`, `inference/`
- Screenshot placeholder: Editor view with project selector

#### Section 2: Project Structure
- Visual file tree showing all 9 card files organized in folders:
  ```
  neural-net-pipeline/
  ├── data/
  │   ├── data_load.py
  │   └── data_split.py
  ├── model/
  │   └── build_model.py
  ├── training/
  │   ├── forward_pass.py
  │   ├── compute_loss.py
  │   ├── backward_pass.py
  │   └── optimizer_step.py
  ├── evaluation/
  │   └── evaluate.py
  └── inference/
      └── inference.py
  ```
- Explanation of folder organization by pipeline stage

#### Section 3: Writing Card Code
- 9 subsections, one per card
- Each subsection has:
  - Card name and file path
  - Brief description of what it does
  - Full Python source code in a `<pre><code>` block
  - Key points about the card's config_schema, input_schema, output_schema
- Cards in order:
  1. Data Load (`data/data_load.py`) — loads CSV from URL
  2. Data Split (`data/data_split.py`) — train/test split
  3. Build Model (`model/build_model.py`) — PyTorch architecture
  4. Forward Pass (`training/forward_pass.py`) — network inference
  5. Compute Loss (`training/compute_loss.py`) — cross-entropy
  6. Backward Pass (`training/backward_pass.py`) — backpropagation
  7. Optimizer Step (`training/optimizer_step.py`) — weight updates
  8. Evaluate (`evaluation/evaluate.py`) — test accuracy
  9. Inference (`inference/inference.py`) — predictions

#### Section 4: Building the Pipeline
- How to switch to Board view
- How to drag cards from the palette
- Connection map table:

  | From | Output | To | Input |
  |------|--------|----|-------|
  | Data Load | dataset | Data Split | dataset |
  | Data Split | train_data | Build Model | train_data |
  | Data Split | test_data | Evaluate | test_data |
  | Build Model | training_state | Forward Pass | training_state |
  | Forward Pass | training_state | Compute Loss | training_state |
  | Compute Loss | training_state | Backward Pass | training_state |
  | Backward Pass | training_state | Optimizer Step | training_state |
  | Optimizer Step | trained_model | Evaluate | trained_model |
  | Optimizer Step | trained_model | Inference | trained_model |

- Visual pipeline diagram:
  ```
  [Data Load] → [Data Split] → [Build Model] → [Forward Pass] → [Compute Loss] → [Backward Pass] → [Optimizer Step] → [Evaluate]
                       ↓                                                                                    ↓
                  [Evaluate]                                                                           [Inference]
  ```

#### Section 5: Configuring Cards
- How to click a card to open config panel
- Default configuration values:
  - Data Load: `source_url` = Iris CSV URL
  - Data Split: `target_column` = "species", `test_ratio` = 0.2
  - Build Model: `hidden_sizes` = "16,8", `learning_rate` = 0.01
  - Optimizer Step: `epochs` = 50
  - Inference: `input_values` = "5.1,3.5,1.4,0.2"

#### Section 6: Running the Pipeline
- Click "Run All" button in the header
- Cards execute in topological order
- Real-time status updates per card
- Click completed cards to view outputs
- Note about auto-save and project persistence

### Guide UI features:
- **Left sidebar navigation** — sticky list of section links, active section highlighted via IntersectionObserver
- **Smooth scroll** — clicking a nav link scrolls to that section smoothly
- **Code blocks** — monospace font, dark background (bg-zinc-900), with card name label
- **Screenshot placeholders** — dashed border boxes with ImageIcon and description text (to be replaced with actual screenshots later)
- **Close** — X button or Escape key

---

## 4. Frontend File Structure

```
src/components/tensorrag/onboarding/
├── onboardingSteps.ts      ← Step data + localStorage key constant
├── guideContent.ts         ← Guide sections + 9 card code strings + connection map
├── OnboardingDialog.tsx    ← Full-screen stepper overlay (5 steps)
└── GuideModal.tsx          ← Full-screen documentation overlay
```

Modified files:
```
src/components/tensorrag/header/TensorRagHeader.tsx   ← Add onOpenGuide prop + BookOpen button
src/app/tensorrag/page.tsx                            ← Wire state, localStorage, render modals
```

---

## 5. Component Architecture

```
page.tsx (PipelineApp)
  state: showOnboarding, showGuide
  │
  ├── TensorRagHeader  (prop: onOpenGuide)
  │   └── BookOpen button → calls props.onOpenGuide()
  │
  ├── OnboardingDialog (props: open, onClose, onOpenGuide)
  │   ├── 5-step stepper with left/right split layout
  │   ├── Navigation: Previous / Next / Get Started
  │   └── "Learn more" link → calls props.onOpenGuide()
  │
  └── GuideModal (props: open, onClose)
      ├── Left sidebar with section nav (IntersectionObserver)
      └── Right content area with scrollable documentation
```

### Props flow:
- `TensorRagHeader` receives `onOpenGuide` callback → triggers `handleOpenOnboarding` in page.tsx
- `OnboardingDialog` receives `open`, `onClose`, `onOpenGuide` → can open guide on top of itself
- `GuideModal` receives `open`, `onClose` → independent overlay at z-[60]

### Z-index layering:
- OnboardingDialog: `z-50` (same level as PremiumLock and other modals)
- GuideModal: `z-[60]` (stacks above OnboardingDialog so both can be open simultaneously)

---

## 6. Data Files Specification

### onboardingSteps.ts

```typescript
// Exports:
export interface OnboardingStep {
  id: string;
  title: string;
  headline: string;
  description: string;
  mediaSrc: string | null;  // null = placeholder, future: GIF/video URL
  mediaAlt: string;
}

export const LOCALSTORAGE_KEY = "tensorrag-onboarding-seen";

export const ONBOARDING_STEPS: OnboardingStep[] = [
  // 5 steps as defined in Section 2
];
```

### guideContent.ts

```typescript
// Exports:
export interface GuideSection {
  id: string;
  title: string;
  content: string;  // markdown-like description for rendering
}

export const GUIDE_SECTIONS: GuideSection[] = [
  // 6 sections as defined in Section 3
];

export interface CardCodeEntry {
  name: string;
  file: string;
  folder: string;
  description: string;
  code: string;  // full Python source
}

export const CARD_CODES: CardCodeEntry[] = [
  // 9 cards with their Python source strings from NEURAL_NETWORK_PIPELINE.md
];

export interface ConnectionEntry {
  from: string;
  output: string;
  to: string;
  input: string;
}

export const CONNECTIONS: ConnectionEntry[] = [
  // 9 connections as defined in Section 3, Section 4
];
```

---

## 7. OnboardingDialog Component Spec

**File:** `OnboardingDialog.tsx`

### Props
```typescript
interface OnboardingDialogProps {
  open: boolean;
  onClose: () => void;
  onOpenGuide: () => void;
}
```

### Internal state
```typescript
const [currentStep, setCurrentStep] = useState(0);
```

### Layout (desktop)
```
┌──────────────────────────────────────────────────────────────────┐
│                                                            [X]  │
│  ┌────────────────────────────┬─────────────────────────────┐   │
│  │  ● ● ● ○ ○               │                              │   │
│  │                            │                              │   │
│  │  Write Cards               │    ┌─ ─ ─ ─ ─ ─ ─ ─ ─ ┐   │   │
│  │                            │    │                     │   │   │
│  │  ┌──────────────────┐      │    │   🎬 Preview        │   │   │
│  │  │  Editor View     │      │    │   coming soon       │   │   │
│  │  └──────────────────┘      │    │                     │   │   │
│  │                            │    └─ ─ ─ ─ ─ ─ ─ ─ ─ ┘   │   │
│  │  Switch to the Editor      │                              │   │
│  │  view to create projects,  │                              │   │
│  │  organize cards...         │                              │   │
│  │                            │                              │   │
│  │  📖 Learn more             │                              │   │
│  │            [Previous] [Next]│                              │   │
│  └────────────────────────────┴─────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

### Styling (follows PremiumLock.tsx pattern)
- **Outer:** `fixed inset-0 z-50 flex items-center justify-center p-4`
- **Backdrop:** `absolute inset-0 bg-black/60 backdrop-blur-sm`
- **Modal:** `relative max-w-4xl w-full bg-zinc-950 border border-zinc-800 rounded-2xl overflow-hidden`
- **Grid:** `grid grid-cols-1 md:grid-cols-2`
- **Left panel:** `p-8` with step dots, title, headline badge, description, nav
- **Right panel:** `bg-zinc-900/30 border-l border-zinc-800 hidden md:flex` with placeholder
- **Step dots:** `w-2 h-2 rounded-full` — active: `bg-purple-500`, inactive: `bg-zinc-700`
- **Headline badge:** `text-purple-400 bg-purple-500/10 px-3 py-1 rounded-full text-xs`
- **Nav buttons:** Previous = `bg-zinc-800`, Next = `bg-purple-600`, Get Started = `bg-emerald-600`
- **Learn more link:** `text-purple-400 hover:text-purple-300` with BookOpen icon

### Behavior
- Uses `createPortal` to `document.body`
- Returns `null` when `open === false`
- Escape key calls `onClose()`
- "Get Started" (last step) calls `onClose()`
- "Learn more" calls `onOpenGuide()`
- Resets `currentStep` to 0 when `open` changes to `true`

---

## 8. GuideModal Component Spec

**File:** `GuideModal.tsx`

### Props
```typescript
interface GuideModalProps {
  open: boolean;
  onClose: () => void;
}
```

### Layout
```
┌──────────────────────────────────────────────────────────────────────┐
│  Getting Started Guide     [neural-net-pipeline]              [X]   │
├────────────┬─────────────────────────────────────────────────────────┤
│            │                                                         │
│  NAV       │  CONTENT (scrollable)                                   │
│  ────      │                                                         │
│  > Creating│  1. Creating a Project                                  │
│    a Proj  │  ─────────────────────                                  │
│            │  Switch to the Editor view...                           │
│  Project   │                                                         │
│  Structure │  ┌─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┐                           │
│            │  │  📷 Screenshot placeholder│                          │
│  Writing   │  └─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┘                           │
│  Card Code │                                                         │
│            │  2. Project Structure                                    │
│  Building  │  ─────────────────────                                  │
│  Pipeline  │  neural-net-pipeline/                                   │
│            │  ├── data/                                               │
│  Config    │  │   ├── data_load.py                                   │
│            │  ...                                                     │
│  Running   │                                                         │
│            │  3. Writing Card Code                                    │
│            │  ─────────────────────                                  │
│            │  3.1 Data Load                                          │
│            │  ┌────────────────────────────┐                         │
│            │  │ from cards.base import...  │                         │
│            │  │ ...                        │                         │
│            │  └────────────────────────────┘                         │
│            │  ...                                                     │
└────────────┴─────────────────────────────────────────────────────────┘
```

### Styling
- **Outer:** `fixed inset-4 z-[60] bg-zinc-950 border border-zinc-800 rounded-2xl flex flex-col overflow-hidden`
- **Header:** `h-14 px-6 flex items-center border-b border-zinc-800 shrink-0`
- **Title:** `text-white font-semibold`
- **Badge:** `text-purple-400 bg-purple-500/10 px-2 py-0.5 rounded text-xs ml-3`
- **Body:** `flex flex-1 overflow-hidden`
- **Nav sidebar:** `w-56 border-r border-zinc-800 p-4 overflow-y-auto hidden md:block`
- **Nav links:** `text-sm py-1.5 px-3 rounded` — active: `text-purple-400 bg-purple-500/10`, inactive: `text-zinc-500 hover:text-zinc-300`
- **Content:** `flex-1 overflow-y-auto p-8`
- **Code blocks:** `bg-zinc-900 border border-zinc-800 rounded-lg p-4 font-mono text-xs text-zinc-300 overflow-x-auto`
- **Screenshot placeholders:** `border-2 border-dashed border-zinc-700 rounded-lg p-8 flex items-center justify-center text-zinc-600`

### Behavior
- Uses `createPortal` to `document.body`
- Returns `null` when `open === false`
- Escape key calls `onClose()`
- Section IDs used for `scrollIntoView({ behavior: 'smooth' })`
- `IntersectionObserver` on section headings updates active nav link
- Code blocks render all 9 card Python sources from `CARD_CODES`
- Connection table renders from `CONNECTIONS`

---

## 9. Header Modification

**File:** `src/components/tensorrag/header/TensorRagHeader.tsx`

### Changes

1. **Add prop to component:**
   ```typescript
   // Before:
   export function TensorRagHeader() {

   // After:
   interface TensorRagHeaderProps {
     onOpenGuide?: () => void;
   }
   export function TensorRagHeader({ onOpenGuide }: TensorRagHeaderProps) {
   ```

2. **Add BookOpen to imports:**
   ```typescript
   // Before:
   import { Play, Loader2, Download, Github, Palette, Bell } from "lucide-react";

   // After:
   import { Play, Loader2, Download, Github, Palette, Bell, BookOpen } from "lucide-react";
   ```

3. **Insert guide button** in the right actions area, before the theme selector:
   ```tsx
   {onOpenGuide && (
     <Tooltip>
       <TooltipTrigger asChild>
         <button onClick={onOpenGuide} className={btnClass}>
           <BookOpen className="h-4 w-4" />
         </button>
       </TooltipTrigger>
       <TooltipContent>Guide</TooltipContent>
     </Tooltip>
   )}
   ```

---

## 10. Page Integration

**File:** `src/app/tensorrag/page.tsx`

### New imports
```typescript
import { useCallback } from "react";
import { OnboardingDialog } from "@/components/tensorrag/onboarding/OnboardingDialog";
import { GuideModal } from "@/components/tensorrag/onboarding/GuideModal";
import { LOCALSTORAGE_KEY } from "@/components/tensorrag/onboarding/onboardingSteps";
```

### New state (inside PipelineApp)
```typescript
const [showOnboarding, setShowOnboarding] = useState(false);
const [showGuide, setShowGuide] = useState(false);
```

### First-visit useEffect
```typescript
useEffect(() => {
  const seen = localStorage.getItem(LOCALSTORAGE_KEY);
  if (!seen) setShowOnboarding(true);
}, []);
```

### Handler functions
```typescript
const handleDismissOnboarding = useCallback(() => {
  localStorage.setItem(LOCALSTORAGE_KEY, "true");
  setShowOnboarding(false);
}, []);

const handleOpenOnboarding = useCallback(() => {
  setShowOnboarding(true);
}, []);

const handleOpenGuide = useCallback(() => {
  setShowGuide(true);
}, []);

const handleCloseGuide = useCallback(() => {
  setShowGuide(false);
}, []);
```

### Pass prop to header
```tsx
<TensorRagHeader onOpenGuide={handleOpenOnboarding} />
```

### Render modals (at end of PipelineApp JSX, before closing `</div>`)
```tsx
<OnboardingDialog
  open={showOnboarding}
  onClose={handleDismissOnboarding}
  onOpenGuide={handleOpenGuide}
/>
<GuideModal
  open={showGuide}
  onClose={handleCloseGuide}
/>
```

---

## 11. State Management

### State diagram
```
                    ┌─────────────────┐
                    │   Page Load      │
                    └────────┬────────┘
                             │
                    Check localStorage
                     "tensorrag-onboarding-seen"
                             │
                    ┌────────┴────────┐
                    │                 │
               NOT found           Found
                    │                 │
           Show Onboarding     Do nothing
                    │
          ┌─────────┴──────────┐
          │                    │
    "Get Started"        "Learn more"
     / X / Escape              │
          │              Show Guide
    Set localStorage      (z-[60])
    Hide Onboarding            │
                         Close Guide
                          (X / Escape)
                               │
                        Return to
                        Onboarding
                        (still open)
```

### Header "BookOpen" button flow
```
Click BookOpen → setShowOnboarding(true) → Dialog opens
                                           (localStorage NOT set,
                                            so closing won't mark as seen
                                            unless "Get Started" is clicked)
```

Wait — correction. The header BookOpen button should show the onboarding dialog. When closed via the dialog (any method), `handleDismissOnboarding` sets localStorage. This is fine for repeat users who want to review.

---

## 12. Media Assets (Future)

Each onboarding step has a `mediaSrc: null` placeholder. In the future:

| Step | Planned Media |
|------|--------------|
| Welcome | Animated overview GIF showing the full pipeline flow |
| Write Cards | Screen recording of creating a card file and writing code |
| Build Pipeline | Screen recording of dragging cards and connecting handles |
| Configure & Run | Screen recording of setting params and clicking Run All |
| View Results | Screen recording of viewing card outputs and metrics |

The guide modal also has screenshot placeholders for:
- Editor view with project selector
- File tree with folders
- Board view with connected cards
- Config panel
- Execution progress
- Output previews

These will be added as static images in `/public/tensorrag/` in a future phase.

---

## 13. Implementation Order

| # | File | Action | Dependencies |
|---|------|--------|-------------|
| 1 | `onboardingSteps.ts` | CREATE | None |
| 2 | `guideContent.ts` | CREATE | None |
| 3 | `OnboardingDialog.tsx` | CREATE | Step 1 |
| 4 | `GuideModal.tsx` | CREATE | Step 2 |
| 5 | `TensorRagHeader.tsx` | MODIFY | None |
| 6 | `page.tsx` | MODIFY | Steps 1-5 |
| 7 | Build check | VERIFY | Step 6 |

Steps 1 and 2 can be done in parallel. Steps 3 and 4 can be done in parallel (after their respective data files). Step 5 is independent. Step 6 requires all previous steps.

---

## 14. Verification Checklist

- [ ] Clear localStorage key `tensorrag-onboarding-seen` and visit `/tensorrag` — onboarding auto-shows
- [ ] Navigate through all 5 steps using Next/Previous buttons
- [ ] Step dots update correctly as you navigate
- [ ] Click "Get Started" on last step — dismisses and sets localStorage
- [ ] Refresh page — onboarding does NOT show again
- [ ] Click BookOpen button in header — onboarding re-opens
- [ ] Click "Learn more" in onboarding — guide modal opens on top (z-[60] above z-50)
- [ ] Close guide modal (X or Escape) — returns to onboarding underneath
- [ ] Escape key closes topmost modal only (guide first, then onboarding)
- [ ] Section nav in guide works (smooth scroll, IntersectionObserver highlights)
- [ ] All 9 card code blocks render correctly in the guide
- [ ] Connection table displays all 9 connections
- [ ] Responsive: stacked layout on mobile, side-by-side on desktop
- [ ] `npm run build` passes with no TypeScript errors
- [ ] No console warnings or errors
