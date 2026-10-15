"""A one-minute explainer of seedling, made with Manim (text and shapes only, no LaTeX).

    pip install manim            # needs cairo + pango; on NixOS: nix-shell -p cairo pango pkg-config
    manim -qh video/explainer.py Explainer       # 1080p, writes video/media/videos/...
"""

from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Arrow,
    Create,
    Cross,
    FadeIn,
    FadeOut,
    GrowArrow,
    Indicate,
    LaggedStart,
    Rectangle,
    RoundedRectangle,
    Scene,
    Text,
    Transform,
    VGroup,
    Write,
    config,
)

BG = "#0f1117"
INK = "#e6e6e6"
MUTED = "#8b93a7"
ACCENT = "#7bd88f"
BLUE = "#6aa9ff"
RED = "#ff6b6b"
GOLD = "#f5c26b"

QWEN_BEFORE = "87%"
QWEN_AFTER = "94%"
REPO = "github.com/4arkeinlvy/seedling"

config.background_color = BG


def label(text: str, size: int = 28, color: str = INK) -> Text:
    return Text(text, font_size=size, color=color)


def box(text: str, width: float = 2.6, color: str = BLUE) -> VGroup:
    frame = RoundedRectangle(corner_radius=0.15, width=width, height=1.0, color=color)
    return VGroup(frame, label(text, 26).move_to(frame))


class Explainer(Scene):
    def construct(self) -> None:
        self.title()
        self.problem()
        self.one_step()
        self.crash()
        self.results()
        self.outro()

    def clear(self) -> None:
        self.play(*[FadeOut(m) for m in self.mobjects])

    def title(self) -> None:
        name = label("seedling", 84, ACCENT)
        tagline = label("train models by sharing seeds, not gradients", 32, MUTED).next_to(name, DOWN)
        self.play(Write(name))
        self.play(FadeIn(tagline, shift=UP * 0.3))
        self.wait(1)
        self.clear()

    def problem(self) -> None:
        heading = label("What travels over the network each step?", 34).to_edge(UP)
        ddp = label("normal training (DDP): the gradients", 28, RED)
        ddp_bar = Rectangle(width=9.0, height=0.5, color=RED, fill_opacity=0.8)
        ddp_note = label("438 KB for a tiny model, ~2 GB for a 0.5B model", 24, MUTED)
        es = label("seedling: seeds + scores", 28, ACCENT)
        es_bar = Rectangle(width=0.08, height=0.5, color=ACCENT, fill_opacity=1.0)
        es_note = label("392 bytes, whatever the model size (bar not to scale: ~1,100x smaller)", 24, MUTED)
        rows = VGroup(ddp, ddp_bar, ddp_note, es, es_bar, es_note).arrange(DOWN, buff=0.3)
        for bar in (ddp_bar, es_bar):
            bar.align_to(rows, LEFT)
        rows.next_to(heading, DOWN, buff=0.6)
        self.play(FadeIn(heading))
        self.play(FadeIn(ddp))
        self.play(Create(ddp_bar), FadeIn(ddp_note))
        self.play(FadeIn(es), Create(es_bar), FadeIn(es_note))
        self.wait(1.5)
        self.clear()

    def one_step(self) -> None:
        heading = label("One training step", 34).to_edge(UP)
        coordinator = box("coordinator", 3.2, GOLD).shift(UP * 1.4)
        workers = VGroup(box("laptop GPU"), box("laptop CPU"), box("Kaggle")).arrange(RIGHT, buff=0.8).shift(DOWN * 1.6)
        self.play(FadeIn(heading), FadeIn(coordinator), LaggedStart(*[FadeIn(w) for w in workers], lag_ratio=0.2))

        down = [Arrow(coordinator.get_bottom(), w.get_top(), color=GOLD, buff=0.1) for w in workers]
        seeds_note = label("seeds", 24, GOLD).next_to(coordinator, RIGHT)
        self.play(*[GrowArrow(a) for a in down], FadeIn(seeds_note))
        try_note = label("each worker tries  theta + noise(seed)  and  theta - noise(seed)", 24, MUTED).next_to(
            workers, DOWN, buff=0.75
        )
        self.play(FadeIn(try_note))
        self.wait(0.8)

        up = [Arrow(w.get_top(), coordinator.get_bottom(), color=ACCENT, buff=0.1) for w in workers]
        scores_note = label("scores", 24, ACCENT).next_to(coordinator, RIGHT)
        self.play(*[Transform(d, u) for d, u in zip(down, up, strict=True)], Transform(seeds_note, scores_note))
        record = label("record: seeds + integer coefficients", 24, GOLD).next_to(coordinator, UP, buff=0.15)
        self.play(FadeIn(record))
        exact = label("state += sum  q x sign(seed)   exact integers, in C", 24, INK).move_to(try_note)
        self.play(Transform(try_note, exact))
        hashes = VGroup(*[label("hash 46443b65", 20, ACCENT).next_to(w, DOWN, buff=0.12) for w in workers])
        self.play(LaggedStart(*[FadeIn(h) for h in hashes], lag_ratio=0.2))
        self.play(*[Indicate(h, color=ACCENT) for h in hashes])
        self.wait(1)
        self.clear()

    def crash(self) -> None:
        heading = label("Machines crash. The math doesn't care.", 34).to_edge(UP)
        workers = VGroup(box("worker 1"), box("worker 2"), box("coordinator", 3.0, GOLD)).arrange(RIGHT, buff=0.6)
        self.play(FadeIn(heading), FadeIn(workers))
        for target in (workers[0], workers[2]):
            kill = Cross(target, stroke_color=RED)
            note = label("kill -9", 26, RED).next_to(target, DOWN)
            self.play(Create(kill), FadeIn(note))
            self.play(FadeOut(kill), FadeOut(note))
        result = label("final weights: bit-identical to a run with no crashes", 28, ACCENT).next_to(
            workers, DOWN, buff=0.8
        )
        self.play(Write(result))
        self.wait(1.5)
        self.clear()

    def results(self) -> None:
        heading = label("Measured on one laptop (RTX 3050, 4 GB)", 34).to_edge(UP)
        rows = VGroup(
            label("CartPole: random policy -> perfect (500) in 14 steps", 28),
            label("MNIST: 87.4% with exact integer ES (backprop 91.7%)", 28),
            label(f"Qwen2.5-0.5B, Countdown: well-formed answers {QWEN_BEFORE} -> {QWEN_AFTER}", 28),
            label("the whole population runs in ONE batched generate()", 26, MUTED),
        ).arrange(DOWN, buff=0.45, aligned_edge=LEFT)
        self.play(FadeIn(heading))
        self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.3) for r in rows], lag_ratio=0.5))
        self.wait(2)
        self.clear()

    def outro(self) -> None:
        name = label("seedling", 72, ACCENT)
        repo = label(REPO, 32, BLUE).next_to(name, DOWN)
        built = label("C, Python, PyTorch, sockets: built from first principles", 26, MUTED).next_to(repo, DOWN)
        self.play(Write(name), FadeIn(repo), FadeIn(built))
        self.wait(2)
