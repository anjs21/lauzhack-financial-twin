"""Parallel: a 2-3 minute presentation (manim + manim-slides). Space or the right arrow advances.

    manim-slides render --quality h presentation/slides.py Parallel   # 1080p60 (--quality l: quick draft)
    manim-slides present Parallel                                  # present from the rendered slides
    manim-slides convert Parallel parallel.html --one-file --offline   # one HTML file for any browser

The numbers in block 4 come from presentation/data.json (python presentation/make_data.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import segno
from manim import *
from manim_slides import Slide

APP_URL = "https://lauzhack-financial-twin.vercel.app/"
DATA = json.loads((Path(__file__).resolve().parent / "data.json").read_text())

X_C = BLUE          # state
U_C = YELLOW        # action
P_C = GREEN         # the world, P
PI_C = PURPLE_A     # the persona, pi
OTHER_C = RED       # everyone else, -i
THETA_C = TEAL      # market tightness
SOFT = GREY_B
DIM = GREY_D
PATH_C = {"work": BLUE, "master": YELLOW, "phd": TEAL}
BLOCKS = ["Model", "World", "Twin", "Lives"]


def caption(text, font_size=30, color=SOFT):
    return Tex(text, font_size=font_size, color=color).to_edge(DOWN, buff=0.45)


def header(i):
    items = VGroup(*[Tex(rf"{k + 1}\; {name}", font_size=26, color=WHITE if k == i else DIM)
                     for k, name in enumerate(BLOCKS)]).arrange(RIGHT, buff=0.8).to_edge(UP, buff=0.3)
    return VGroup(items, Underline(items[i], color=WHITE, stroke_width=2, buff=0.08))


def label(mob, text, direction, color=SOFT, font_size=28):
    brace = Brace(mob, direction, buff=0.12, color=color)
    return VGroup(brace, Tex(text, font_size=font_size, color=color).next_to(brace, direction, buff=0.12))


def box(text, color=SOFT, font_size=32, pad=(0.6, 0.45)):
    t = Tex(text, font_size=font_size)
    frame = RoundedRectangle(corner_radius=0.15, width=t.width + pad[0], height=t.height + pad[1],
                             stroke_color=color, stroke_width=2)
    return VGroup(frame, t)


class Parallel(Slide):
    skip_reversing = True          # no backward-playback videos: they are not needed and slow to build

    def construct(self):
        self.title()
        self.hook()
        self.model()
        self.world()
        self.twin()
        self.lives()
        self.close()

    def clear(self, keep=()):
        gone = [m for m in self.mobjects if m not in keep]
        if gone:
            self.play(*[FadeOut(m) for m in gone])

    # ------------------------------------------------------------------ opening
    def title(self):
        name = Tex(r"Parallel", font_size=110)
        sub = Tex(r"one choice, thousands of lives", font_size=42, color=SOFT).next_to(name, DOWN, buff=0.4)
        foot = Tex(r"LauzHack 2026 \quad$\cdot$\quad UBS AI Financial Twin", font_size=26, color=GREY_C).to_edge(DOWN, buff=0.6)
        self.play(Write(name), run_time=1.5)
        self.play(FadeIn(sub, shift=0.2 * UP), FadeIn(foot))
        self.next_slide()

    def hook(self):
        self.clear()
        who = Tex(r"Jaycent, 22, from Indonesia", font_size=54)
        what = Tex(r"finishing a bachelor's degree, choosing between", font_size=34, color=SOFT)
        opts = VGroup(*[box(t) for t in (r"PhD in Australia", r"Master's in China", r"MSc in Europe")])
        opts.arrange(RIGHT, buff=0.5)
        question = Tex(r"Which life does each choice lead to?", font_size=40, color=U_C)
        VGroup(who, what, opts, question).arrange(DOWN, buff=0.55)
        self.play(Write(who))
        self.play(FadeIn(what))
        self.play(LaggedStart(*[FadeIn(o, shift=0.2 * UP) for o in opts], lag_ratio=0.25))
        self.play(Write(question))
        self.next_slide()

    # ------------------------------------------------------------------ 1. the model
    def model(self):
        self.clear()
        self.head = header(0)
        self.play(FadeIn(self.head))

        mdp = MathTex(r"x_{t+1}", r"\sim", r"P", r"(\,\cdot \mid", r"x_t", r",\,", r"u_t", r")", font_size=84)
        mdp[0].set_color(X_C), mdp[4].set_color(X_C), mdp[6].set_color(U_C), mdp[2].set_color(P_C)
        notes = VGroup(label(mdp[0], r"next month", DOWN, X_C), label(mdp[4], r"this month", DOWN, X_C),
                       label(mdp[2], r"the world, with noise", UP, P_C), label(mdp[6], r"your choice", UP, U_C))
        top = max(notes[2][1].get_y(), notes[3][1].get_y())
        notes[2][1].set_y(top), notes[3][1].set_y(top)
        cap = caption(r"A Markov decision process: one step per month, 2025 to 2034")
        self.play(Write(mdp))
        self.play(LaggedStart(*[FadeIn(n) for n in notes], lag_ratio=0.3), FadeIn(cap))
        self.next_slide()

        # the state
        small = mdp.copy().scale(0.5).next_to(self.head, DOWN, buff=0.35)
        self.play(FadeOut(notes), FadeOut(cap), Transform(mdp, small))
        names = ["W", "Y", "H", "R", "F", "L", "M", "C", "I"]
        words = ["work", "income", "housing", "rent", "household", "lifestyle", "economy", "cash", "investments"]
        vec = Matrix([[n] for n in names], v_buff=0.56, bracket_h_buff=0.25).scale(0.8)
        vec.get_entries().set_color(X_C)
        lhs = MathTex(r"x_t", r"=", font_size=56).next_to(vec, LEFT, buff=0.25)
        lhs[0].set_color(X_C)
        tags = VGroup()
        for entry, word in zip(vec.get_entries(), words):
            tags.add(Tex(word, font_size=28, color=SOFT).next_to(vec, RIGHT, buff=0.45).set_y(entry.get_y()))
        for t in tags:
            t.align_to(tags[0], LEFT)
        groups = VGroup(
            label(VGroup(*tags[:7]), r"situation", RIGHT, SOFT),
            label(VGroup(*tags[7:]), r"money", RIGHT, SOFT))
        state = VGroup(lhs, vec, tags, groups).move_to(0.3 * DOWN)
        cap = caption(r"9 variables, chosen with correlation tests and importance analysis")
        self.play(FadeIn(lhs), Create(vec.get_brackets()))
        self.play(LaggedStart(*[AnimationGroup(Write(e), FadeIn(t, shift=0.1 * LEFT))
                                for e, t in zip(vec.get_entries(), tags)], lag_ratio=0.12))
        self.play(FadeIn(groups), FadeIn(cap))
        self.next_slide()

        # the action
        self.play(FadeOut(state), FadeOut(cap))
        act = MathTex(r"u_t", r"=", r"(", r"\text{work}", r",\;", r"\text{housing}", r",\;", r"\text{lifestyle}",
                      r",\;", r"\text{savings}", r")", font_size=56).shift(1.7 * UP)
        act[0].set_color(U_C)
        heads = [act[3], act[5], act[7], act[9]]
        for h in heads:
            h.set_color(U_C)
        verbs = [["continue", "enrol", "switch hours", "seek a job", "quit"], ["downsize", "stay", "upsize"],
                 ["cut", "keep", "raise"], ["sell", "hold", "invest"]]
        cols = VGroup()
        for h, vs in zip(heads, verbs):
            col = VGroup(*[Tex(v, font_size=28, color=SOFT) for v in vs]).arrange(DOWN, buff=0.18)
            cols.add(col.next_to(act, DOWN, buff=0.5).set_x(h.get_x()))
        size = MathTex(r"|\mathcal{A}| = 5 \cdot 3 \cdot 3 \cdot 3 = 135", font_size=34, color=SOFT)
        size.next_to(cols, DOWN, buff=0.5)
        cap = caption(r"Each January, four choices; the state decides which ones are allowed")
        self.play(Write(act))
        self.play(LaggedStart(*[FadeIn(c, shift=0.15 * DOWN) for c in cols], lag_ratio=0.2))
        self.play(FadeIn(size), FadeIn(cap))
        self.next_slide()

    # ------------------------------------------------------------------ 2. the world
    def world(self):
        self.clear(keep=[self.head])
        self.play(Transform(self.head, header(1)))

        def P(*rest):
            parts = [r"P", r"(", r"x_i'", r"\mid", r"x_i", r",\,", r"u_i", *rest, r")"]
            m = MathTex(*parts, font_size=72)
            m[2].set_color(X_C), m[4].set_color(X_C), m[6].set_color(U_C), m[0].set_color(P_C)
            return m

        own = P()
        full = P(r",\,", r"x_{-i}", r",\,", r"u_{-i}")
        full[8].set_color(OTHER_C), full[10].set_color(OTHER_C)
        self.play(Write(own))
        self.wait(0.4)
        self.play(TransformMatchingTex(own, full))
        others = label(VGroup(full[8], full[10]), r"everyone else", DOWN, OTHER_C)
        cap = caption(r"Your chance of a job offer depends on who else applies")
        self.play(FadeIn(others), FadeIn(cap))
        self.next_slide()

        # a society competing for a few openings
        self.play(FadeOut(others), FadeOut(cap), full.animate.scale(0.7).next_to(self.head, DOWN, buff=0.4))
        rng = np.random.default_rng(3)
        dots = VGroup(*[Dot(radius=0.075, color=GREY_C) for _ in range(14 * 6)]).arrange_in_grid(6, 14, buff=0.28)
        dots.move_to(2.6 * LEFT + 0.6 * DOWN)
        applicants = rng.choice(len(dots), 24, replace=False)
        openings = VGroup(*[Square(0.32, color=THETA_C, fill_opacity=0.25) for _ in range(4)]).arrange(DOWN, buff=0.3)
        openings.move_to(2.3 * RIGHT + 0.6 * DOWN)
        open_tag = Tex(r"openings", font_size=26, color=THETA_C).next_to(openings, UP, buff=0.2)
        targets = rng.integers(0, 4, len(applicants))
        links = VGroup(*[Line(dots[a].get_center(), openings[k].get_left(), stroke_width=1, color=GREY_D)
                         for a, k in zip(applicants, targets)])
        links.set_z_index(-1)
        winners = [applicants[np.flatnonzero(targets == k)[0]] for k in range(4) if (targets == k).any()]
        cap = VGroup(Tex(r"195 simulated societies $\cdot$ 90{,}000 agents $\cdot$ 10 years", font_size=30),
                     Tex(r"competing for limited jobs, study places and flats", font_size=30, color=SOFT))
        cap.arrange(DOWN, buff=0.15).to_edge(DOWN, buff=0.35)
        self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in dots], lag_ratio=0.01), FadeIn(cap[0]))
        self.play(*[dots[a].animate.set_color(WHITE) for a in applicants], FadeIn(openings), FadeIn(open_tag))
        self.play(LaggedStart(*[Create(l) for l in links], lag_ratio=0.04), FadeIn(cap[1]))
        self.play(*[dots[w].animate.set_color(THETA_C) for w in winners],
                  *[dots[a].animate.set_color(OTHER_C) for a in applicants if a not in winners])
        self.next_slide()

        # tightness
        theta = MathTex(r"\theta", r"=", r"\frac{\text{openings}}{\text{applications}}", r"=", r"\frac{4}{24}",
                        font_size=40).move_to(5.0 * RIGHT + 0.6 * DOWN)
        theta[0].set_color(THETA_C)
        theta.scale_to_fit_width(min(theta.width, 3.6)).set_x(5.05)
        tcap = caption(r"One number sums up everyone else: the market's tightness $\theta$")
        self.play(FadeOut(cap), Write(theta), FadeIn(tcap))
        self.next_slide()

        # mean field: remove the conditioning on -i
        self.play(FadeOut(VGroup(dots, openings, open_tag, links, theta, tcap)),
                  full.animate.scale(1 / 0.7).move_to(1.3 * UP))
        meanf = P(r",\,", r"\theta")
        meanf[8].set_color(THETA_C)
        meanf.move_to(full)
        cap = caption(r"Mean field: everyone else enters only through $\theta$")
        self.play(ReplacementTransform(VGroup(*full[:8]), VGroup(*meanf[:8])),
                  ReplacementTransform(VGroup(full[8], full[9], full[10]), meanf[8]),
                  ReplacementTransform(full[11], meanf[9]), FadeIn(cap))
        avg = MathTex(r"P(x_i' \mid x_i, u_i)", r"=", r"\mathbb{E}_{\theta \mid M}", r"\big[\, P(x_i' \mid x_i, u_i,",
                      r"\theta", r")\, \big]", font_size=56).next_to(meanf, DOWN, buff=0.9)
        avg[4].set_color(THETA_C)
        avg[0].set_color(P_C)
        cap2 = caption(r"$\theta$ is not observed: average over its distribution in the economy $M$")
        self.wait(0.3)
        self.play(Write(avg), ReplacementTransform(cap, cap2))
        self.next_slide()

        # calibration
        self.play(FadeOut(VGroup(meanf, avg, cap2)))
        learned = MathTex(r"P(\text{job offer})", r"\qquad", r"P(\text{admitted})", r"\qquad", r"P(\text{flat})",
                          font_size=48).shift(1.3 * UP)
        for k in (0, 2, 4):
            learned[k].set_color(P_C)
        src = Tex(r"learned from 523{,}000 simulated applications, in a society", font_size=32, color=SOFT)
        src2 = Tex(r"calibrated to Swiss statistics (SECO, BFS)", font_size=32, color=SOFT)
        rows = VGroup(
            Tex(r"bachelor's $\to$ master's: \; simulated 84\% \quad BFS 87\%", font_size=32),
            Tex(r"master's graduates unemployed after a year: \; simulated 6.0\% \quad BFS 6.4\%", font_size=32),
            Tex(r"4 of 6 Swiss targets within their band", font_size=28, color=SOFT))
        VGroup(src, src2).arrange(DOWN, buff=0.15).next_to(learned, DOWN, buff=0.6)
        rows.arrange(DOWN, buff=0.25).next_to(src2, DOWN, buff=0.6)
        self.play(Write(learned))
        self.play(FadeIn(src), FadeIn(src2))
        self.play(LaggedStart(*[FadeIn(r, shift=0.1 * UP) for r in rows], lag_ratio=0.3))
        self.next_slide()

    # ------------------------------------------------------------------ 3. the twin
    def twin(self):
        self.clear(keep=[self.head])
        self.play(Transform(self.head, header(2)))

        pol = MathTex(r"u_t", r"\sim", r"\pi", r"(\,\cdot \mid", r"x_t", r",\,", r"h_t", r",\,", r"z", r")",
                      font_size=76).shift(1.6 * UP)
        pol[0].set_color(U_C), pol[2].set_color(PI_C), pol[4].set_color(X_C), pol[6].set_color(GREY_A)
        pol[8].set_color(PI_C)
        notes = VGroup(label(pol[6], r"your path so far", UP, GREY_A), label(pol[8], r"your traits", DOWN, PI_C))
        traits = ["risk tolerance", "frugality", "career drive", "free time", "housing aspiration", "inertia",
                  "impatience", "goal focus"]
        values = [0.45, 0.6, 0.85, 0.35, 0.5, 0.55, 0.4, 0.7]
        bars = VGroup()
        for k, (name, v) in enumerate(zip(traits, values)):
            col, row = divmod(k, 4)
            track = Rectangle(width=1.8, height=0.14, stroke_width=0, fill_color=GREY_E, fill_opacity=1)
            track.move_to([-1.6 + 5.6 * col, -0.55 * row, 0])
            fill = Rectangle(width=1.8 * v, height=0.14, stroke_width=0, fill_color=PI_C, fill_opacity=1)
            fill.align_to(track, LEFT).set_y(track.get_y())
            tag = Tex(name, font_size=24, color=SOFT).next_to(track, LEFT, buff=0.3)
            bars.add(VGroup(tag, VGroup(track, fill)))
        bars.move_to(1.75 * DOWN)
        cap = caption(r"Sampled, never optimised: it decides the way you tend to decide")
        self.play(Write(pol))
        self.play(FadeIn(notes))
        self.play(LaggedStart(*[AnimationGroup(FadeIn(b[0]), FadeIn(b[1][0]), GrowFromEdge(b[1][1], LEFT))
                                for b in bars], lag_ratio=0.1), FadeIn(cap))
        self.next_slide()

        # the planner agent
        self.play(FadeOut(VGroup(pol, notes, bars, cap)))
        words = box(r"your own\\words", SOFT, 30)
        plan = box(r"LLM planner:\\a 10-year plan", PI_C, 30)
        sim = box(r"300 simulated\\lives", P_C, 30)
        signed = box(r"signed\\plan", WHITE, 30)
        chain = VGroup(words, plan, sim, signed).arrange(RIGHT, buff=0.7).shift(0.6 * UP)
        arrows = VGroup(*[Arrow(a.get_right(), b.get_left(), buff=0.1, stroke_width=3, color=SOFT,
                                max_tip_length_to_length_ratio=0.2) for a, b in zip(chain[:-1], chain[1:])])
        back = CurvedArrow(sim.get_bottom() + 0.05 * DOWN, plan.get_bottom() + 0.05 * DOWN, angle=-TAU / 3,
                           color=SOFT, stroke_width=3)
        back_tag = Tex(r"revise", font_size=28, color=SOFT).next_to(back, DOWN, buff=0.1)
        cap = VGroup(Tex(r"A persona with your traits, plus an LLM planner that writes your own plan", font_size=30),
                     Tex(r"plans change? nudge it, and it revises only what changed", font_size=30, color=SOFT))
        cap.arrange(DOWN, buff=0.15).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(words, shift=0.1 * RIGHT))
        self.play(GrowArrow(arrows[0]), FadeIn(plan))
        self.play(GrowArrow(arrows[1]), FadeIn(sim))
        self.play(Create(back), FadeIn(back_tag))
        self.play(GrowArrow(arrows[2]), FadeIn(signed), FadeIn(cap))
        self.next_slide()

    # ------------------------------------------------------------------ 4. the lives
    def lives(self):
        self.clear(keep=[self.head])
        self.play(Transform(self.head, header(3)))

        age = np.array(DATA["age"])
        low = -150                                   # the x-axis sits at the bottom of the chart, not at zero
        title = Tex(r"Jaycent: 2{,}000 simulated lives per choice", font_size=36).next_to(self.head, DOWN, buff=0.3)
        ax = Axes(x_range=[22, 32, 2], y_range=[0, 600, 100], x_length=6.2, y_length=4.2,
                  axis_config={"color": GREY_C, "stroke_width": 2, "include_tip": False, "font_size": 24},
                  x_axis_config={"numbers_to_include": range(22, 33, 2)},
                  y_axis_config={"include_ticks": False})
        ax.to_edge(LEFT, buff=1.5).shift(0.1 * UP)

        def c2p(a, y):
            return ax.c2p(a, y - low)

        y_nums = VGroup(*[MathTex(str(v), font_size=24, color=GREY_B).next_to(c2p(22, v), LEFT, buff=0.15)
                          for v in range(-100, 401, 100)])
        x_tag = Tex(r"age", font_size=26, color=SOFT).next_to(ax.x_axis, RIGHT, buff=0.25)
        y_tag = Tex(r"net wealth, CHF thousands", font_size=26, color=SOFT).rotate(PI / 2)
        y_tag.next_to(y_nums, LEFT, buff=0.3)
        zero = DashedLine(c2p(22, 0), c2p(32, 0), color=GREY_D, stroke_width=1.5)

        def curve(ys, color, width=3, opacity=1.0):
            return VMobject(stroke_color=color, stroke_width=width, stroke_opacity=opacity).set_points_as_corners(
                [c2p(a, y) for a, y in zip(age, ys)])

        def band(path):
            f = DATA["paths"][path]["fan_k"]
            pts = [c2p(a, y) for a, y in zip(age, f["p25"])] + [c2p(a, y) for a, y in zip(age[::-1], f["p75"][::-1])]
            return Polygon(*pts, stroke_width=0, fill_color=PATH_C[path], fill_opacity=0.16)

        self.play(FadeIn(title), Create(ax), FadeIn(y_nums), FadeIn(x_tag), FadeIn(y_tag), FadeIn(zero))
        inside = [s for s in DATA["paths"]["work"]["sample_lives_k"] if low <= min(s) and max(s) <= 450][:14]
        lives = VGroup(*[curve(s, PATH_C["work"], 1.2, 0.45) for s in inside])
        self.play(LaggedStart(*[Create(l) for l in lives], lag_ratio=0.08, run_time=2))

        legend = VGroup()
        for path in ("work", "master", "phd"):
            name = {"work": r"Work now", "master": r"Master's", "phd": r"PhD"}[path]
            legend.add(VGroup(Line(ORIGIN, 0.45 * RIGHT, color=PATH_C[path], stroke_width=5),
                              Tex(name, font_size=30)).arrange(RIGHT, buff=0.2))
        legend.arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to([1.6, 1.1, 0], aligned_edge=LEFT)
        mid = curve(DATA["paths"]["work"]["fan_k"]["p50"], PATH_C["work"], 4)
        cap = caption(r"Thin lines: single lives. Band: the middle half. Bold line: the median.", 26)
        self.play(FadeOut(lives), FadeIn(band("work")), Create(mid), FadeIn(legend[0]), FadeIn(cap))
        self.next_slide()

        for path, row in (("master", legend[1]), ("phd", legend[2])):
            self.play(FadeIn(band(path)), Create(curve(DATA["paths"][path]["fan_k"]["p50"], PATH_C[path], 4)),
                      FadeIn(row))
        self.next_slide()

        # the numbers
        p = DATA["paths"]
        xs = (4.75, 6.2)
        cols = VGroup(Tex(r"net wealth\\at 32", font_size=22, color=SOFT),
                      Tex(r"first job\\at age", font_size=22, color=SOFT))
        for col, x in zip(cols, xs):
            col.move_to([x, legend.get_top()[1] + 0.5, 0])
        table = VGroup()
        for row, path in zip(legend, ("work", "master", "phd")):
            vals = VGroup(Tex(rf"CHF {p[path]['median_net_wealth_at_32_k']:.0f}k", font_size=28),
                          Tex(rf"{p[path]['median_age_first_job']:.1f}", font_size=28))
            for v, x in zip(vals, xs):
                v.move_to([x, row.get_y(), 0])
            table.add(vals)
        note = Tex(r"medians over 2{,}000 lives; " + rf"{p['phd']['share_with_phd'] * 100:.0f}\% finish the PhD",
                   font_size=22, color=SOFT).next_to(legend, DOWN, buff=0.45).align_to(legend, LEFT)
        foot = VGroup(
            Tex(r"Illustrative start: finishing a bachelor's in Lausanne, CHF 8{,}000 saved.", font_size=20),
            Tex(r"The PhD path includes the master's the model requires.", font_size=20),
            Tex(r"Swiss-calibrated world (CHF): countries and currencies are the next extension.", font_size=20))
        foot.set_color(GREY_C).arrange(DOWN, buff=0.06).to_edge(DOWN, buff=0.2)
        self.play(FadeOut(cap), FadeIn(cols), LaggedStart(*[FadeIn(v) for v in table], lag_ratio=0.25),
                  FadeIn(note), FadeIn(foot))
        self.next_slide()

    # ------------------------------------------------------------------ close
    def close(self):
        self.clear()
        a = Tex(r"Don't predict the future.", font_size=52)
        b = Tex(r"Make it explorable.", font_size=52, color=U_C)
        tag = VGroup(Tex(r"Parallel scales with computation:", font_size=26, color=SOFT),
                     Tex(r"more people, more markets, more countries", font_size=26, color=SOFT))
        tag.arrange(DOWN, buff=0.12)
        left = VGroup(a, b, tag).arrange(DOWN, buff=0.35).move_to(3.0 * LEFT)
        tag.shift(0.5 * DOWN)

        # QR code to the app, black modules on a white card with a quiet zone so phones can scan it
        matrix = segno.make(APP_URL, error="m").matrix
        n, cell = len(matrix), 0.085
        card = Square((n + 4) * cell, stroke_width=0, fill_color=WHITE, fill_opacity=1)
        modules = VGroup()          # one block per horizontal run, overlapping a little: no anti-aliasing seams
        for i, row in enumerate(matrix):
            j = 0
            while j < n:
                if row[j]:
                    k = j
                    while k < n and row[k]:
                        k += 1
                    block = Rectangle(width=(k - j) * cell, height=cell * 1.08, stroke_width=0,
                                      fill_color=BLACK, fill_opacity=1)
                    modules.add(block.move_to(card.get_corner(UL) + ((2 + (j + k) / 2) * cell) * RIGHT
                                              + ((2 + i + 0.5) * cell) * DOWN))
                    j = k
                else:
                    j += 1
        qr = VGroup(card, modules)
        try_it = Tex(r"Try it", font_size=34)
        url = Tex(r"lauzhack-financial-twin.vercel.app", font_size=24, color=SOFT)
        right = VGroup(try_it, qr, url).arrange(DOWN, buff=0.25).move_to(4.0 * RIGHT)
        self.play(Write(a))
        self.play(Write(b))
        self.play(FadeIn(tag), FadeIn(right))
        self.next_slide()
