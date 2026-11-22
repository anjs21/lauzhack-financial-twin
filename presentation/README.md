# Presentation: Parallel in 2–3 minutes

A 16-step presentation made with [manim](https://www.manim.community/) and
[manim-slides](https://manim-slides.eertmans.be/). Each press of **space** (or the right arrow) plays
the next step. There are four blocks: 1 Model, 2 World, 3 Twin, 4 Lives.

| File | What |
|---|---|
| `slides.py` | The presentation (one `Slide` scene, `Parallel`) |
| `make_data.py` | Simulates the example's three paths and writes `data.json` (~1 min) |
| `data.json` | The numbers and curves shown in block 4 |
| `parallel.html` | The rendered presentation in one file (not in git: build it below) |

## Present

Open `parallel.html` in any browser and press space. It works offline.

## Build

Needs manim 0.19 (manim-slides 5.7 does not support manim 0.22 yet), manim-slides and a LaTeX
installation with `standalone`, `preview`, `doublestroke`, `setspace`, `rsfs`, `relsize`,
`ragged2e`, `everysel`, `microtype`, `dvisvgm`, `amsfonts`, `babel-english`, `cm-super`.

```bash
conda create -p ~/envs/manim -c conda-forge python=3.12 "manim=0.19.1" && ~/envs/manim/bin/pip install manim-slides segno
cd presentation
python make_data.py                                            # only if the model changed
manim-slides render --quality h slides.py Parallel             # 1080p60; --quality l for a quick draft (not -qh: manim-slides reads it as -h)
manim-slides convert Parallel parallel.html --one-file --offline
manim-slides present Parallel                                  # alternative: a native window (needs Qt)
```

## About the numbers in block 4

* Jaycent's three options are simulated as the model can represent them: **PhD vs master's vs
  starting work**. The model is calibrated to Switzerland and has no countries or currencies, so it
  cannot compare Australia, China and Europe; the slide says so.
* The game-theory transition model
  ([`twin_game_theory/world/transition.py`](../twin_game_theory/world/transition.py)) runs the paths,
  because it is the one with a PhD market. In it, a PhD requires a master's, so the PhD path includes
  one.
* Illustrative start: finishing a bachelor's in Lausanne with CHF 8,000 saved. The decision rules set
  only the study choice; housing, spending and savings stay as they are.
* In 93–100% of these lives cash crosses the student credit limit (3 × monthly pay). This is the
  known student-limit issue (see [`pipeline/09-limitations.md`](../pipeline/09-limitations.md)),
  made worse because these rules never cut spending. The slide does not show it.
