"""
Genera catan_ai_ppo.pdf — documentación de la Fase 4 (PPO).
Ejecutar: python generate_ppo_pdf.py
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Preformatted, KeepTogether,
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT

doc = SimpleDocTemplate(
    "instrucciones/catan_ai_ppo.pdf",
    pagesize=A4,
    leftMargin=2.5*cm, rightMargin=2.5*cm,
    topMargin=2.2*cm,  bottomMargin=2.2*cm,
    title="Catan AI — Fase 4: PPO",
    author="Catan AI Project",
)

W = A4[0] - 5*cm

# ─── Estilos ──────────────────────────────────────────────────────────────────

def sty(name, **kw):
    return ParagraphStyle(name, **kw)

BODY   = sty("body",  fontName="Helvetica",            fontSize=9.5, leading=14, spaceAfter=5,  alignment=TA_JUSTIFY)
H1     = sty("h1",    fontName="Helvetica-Bold",        fontSize=15, leading=19, spaceBefore=14, spaceAfter=6,  textColor=colors.HexColor("#1a3a6b"))
H2     = sty("h2",    fontName="Helvetica-Bold",        fontSize=11, leading=15, spaceBefore=9,  spaceAfter=4,  textColor=colors.HexColor("#2c5f8a"))
TITLE  = sty("title", fontName="Helvetica-Bold",        fontSize=26, leading=32, alignment=TA_CENTER, textColor=colors.HexColor("#1a3a6b"))
SUBT   = sty("subt",  fontName="Helvetica",             fontSize=13, leading=17, alignment=TA_CENTER, textColor=colors.HexColor("#2c5f8a"), spaceAfter=3)
CENTER = sty("ctr",   fontName="Helvetica",             fontSize=9.5, leading=14, alignment=TA_CENTER)
MATH   = sty("math",  fontName="Courier-Bold",          fontSize=10,  leading=14, alignment=TA_CENTER, spaceAfter=6, spaceBefore=4)
BULLET = sty("bul",   fontName="Helvetica",             fontSize=9.5, leading=13, leftIndent=14, spaceAfter=3)
CODE_S = sty("code",  fontName="Courier",               fontSize=8,   leading=11, leftIndent=8,
             backColor=colors.HexColor("#f5f5f5"), spaceAfter=5)

def h1(t):    return Paragraph(t, H1)
def h2(t):    return Paragraph(t, H2)
def p(t):     return Paragraph(t, BODY)
def math(t):  return Paragraph(t, MATH)
def sp(n=5):  return Spacer(1, n)
def hr():     return HRFlowable(width="100%", thickness=0.5,
                                color=colors.HexColor("#cccccc"), spaceAfter=6, spaceBefore=2)
def code(t):  return Preformatted(t.strip(), CODE_S, maxLineLength=95)
def bullets(items): return [Paragraph(f"• {i}", BULLET) for i in items]

def tabla(data, col_widths=None):
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("FONTNAME",  (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",  (0,0), (-1,-1), 8.5),
        ("BACKGROUND",(0,0), (-1,0),  colors.HexColor("#1a3a6b")),
        ("TEXTCOLOR", (0,0), (-1,0),  colors.white),
        ("ALIGN",     (0,0), (-1,-1), "CENTER"),
        ("ALIGN",     (0,1), (1,-1),  "LEFT"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#f0f4fa"), colors.white]),
        ("GRID",      (0,0), (-1,-1), 0.4, colors.HexColor("#bbbbbb")),
        ("TOPPADDING",(0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("LEFTPADDING", (0,0), (-1,-1), 7),
    ]))
    return t

# ─── Historia ─────────────────────────────────────────────────────────────────

story = []

# ── Encabezado ────────────────────────────────────────────────────────────────
story += [
    Paragraph("Catan AI", TITLE),
    sp(6),
    Paragraph("Fase 4 — Proximal Policy Optimization (PPO)", SUBT),
    Paragraph("Actor-Critic · Clipped Surrogate Loss · Entrenamiento por Batches", SUBT),
    sp(10),
    HRFlowable(width="70%", thickness=1, color=colors.HexColor("#1a3a6b"), hAlign="CENTER"),
    sp(10),
]

# ── 1. Por qué PPO ────────────────────────────────────────────────────────────
story += [
    h1("1. De REINFORCE a PPO"),
    hr(),
    p("REINFORCE (Fase 3) logró un <b>70 % de victorias</b> contra el heurístico. "
      "Sin embargo, tiene dos limitaciones estructurales que PPO resuelve:"),
    sp(3),
    tabla(
        [["Problema en REINFORCE", "Solución en PPO"],
         ["Alta varianza: el gradiente usa la recompensa cruda (+1/−1) sin baseline",
          "Critic V(s): estima el valor de cada estado; la ventaja A = G − V(s) es más precisa"],
         ["Una sola actualización por partida: datos usados una vez y descartados",
          "Batch de N partidas + K épocas: cada experiencia se reutiliza K veces"],
         ["Sin límite en el cambio de política: una mala partida puede destruir lo aprendido",
          "Clipping: el ratio π_nueva/π_vieja se recorta a [1−ε, 1+ε]"]],
        col_widths=[7.5*cm, 8*cm],
    ),
    sp(4),
]

# ── 2. Arquitectura Actor-Critic ──────────────────────────────────────────────
story += [
    h1("2. Arquitectura: CatanActorCritic"),
    hr(),
    p("Se extiende <font name='Courier'>CatanNet</font> añadiendo una <b>cabeza de valor</b> "
      "compartiendo el tronco con la cabeza de política:"),
    sp(4),
    math("estado (250)  →  Trunk [Linear(256)→ReLU] × 3  →  h (256)"),
    math("h  →  Policy head  →  logits (249)     [actor]"),
    math("h  →  Value head   →  V(s) ∈ ℝ         [crítico]"),
    sp(4),
    code("""\
class CatanActorCritic(nn.Module):
    def __init__(self, hidden_size=256, num_layers=3):
        self.trunk       = Sequential([Linear(250,256), ReLU] × 3)   # compartido
        self.policy_head = Linear(256, 249)                           # actor
        self.value_head  = Linear(256, 1)                             # crítico

    def forward(self, x):
        h = self.trunk(x)
        return self.policy_head(h), self.value_head(h)   # logits, V(s)"""),
    sp(2),
    p("El método <font name='Courier'>load_from_reinforce()</font> transfiere el tronco y la "
      "cabeza de política desde un checkpoint de Fase 3, arrancando con el conocimiento ya "
      "adquirido. La cabeza de valor se inicializa aleatoriamente."),
    sp(4),
]

# ── 3. Algoritmo PPO ──────────────────────────────────────────────────────────
story += [
    h1("3. Algoritmo PPO-Clip"),
    hr(),
    h2("3.1 Recolección del batch"),
    p("Se juegan <b>N = 16 partidas</b> con la política actual. Para cada paso de decisión "
      "se guarda (sₜ, aₜ, log π_old(aₜ|sₜ), máscara de acciones válidas)."),
    sp(3),

    h2("3.2 Retorno descontado"),
    p("Con recompensa terminal R ∈ {+1, −1} y factor de descuento γ = 0.99:"),
    math("Gₜ  =  γ^(T−1−t) · R"),
    p("Todos los pasos de la misma partida heredan el mismo resultado, pero las decisiones "
      "anteriores pesan menos por el descuento."),
    sp(3),

    h2("3.3 Ventaja normalizada"),
    math("Aₜ  =  Gₜ − V(sₜ)          →          Aₜ ← (Aₜ − μ) / (σ + ε)"),
    p("El crítico V(sₜ) actúa como <i>baseline</i>: si la partida termina con R=+1 "
      "pero el crítico ya esperaba +0.8, la ventaja real es solo +0.2 en ese estado."),
    sp(3),

    h2("3.4 Pérdida total (K = 4 épocas por batch)"),
    p("Para cada época se recomputan los logits y valores con la política <i>actual</i>:"),
    math("rₜ  =  exp( log π_nueva(aₜ|sₜ) − log π_old(aₜ|sₜ) )"),
    math("L_clip  =  −E[ min( rₜ · Aₜ ,  clip(rₜ, 1−ε, 1+ε) · Aₜ ) ]"),
    math("L_value =  MSE( V(sₜ) , Gₜ )"),
    math("L_ent   =  −H[π]      (bonificación de entropía → exploración)"),
    math("L_total =  L_clip  +  0.5 · L_value  −  0.01 · L_ent"),
    sp(4),
]

# ── 4. Hiperparámetros ────────────────────────────────────────────────────────
story += [
    KeepTogether([
        h1("4. Hiperparámetros"),
        hr(),
        tabla(
            [["Parámetro",          "Valor",  "Descripción"],
             ["Learning rate",      "3×10⁻⁴", "Adam optimizer"],
             ["Batch size",         "16",      "Episodios por actualización"],
             ["PPO epochs",         "4",       "Actualizaciones por batch"],
             ["Clip ε",             "0.2",     "Límite del ratio π_nueva/π_old"],
             ["Descuento γ",        "0.99",    "Peso del futuro"],
             ["Value coef",         "0.5",     "Peso de L_value en la pérdida total"],
             ["Entropy coef",       "0.01",    "Peso del bonus de entropía"],
             ["Max grad norm",      "0.5",     "Clipping de gradiente"],
             ["Hidden size",        "256",     "Neuronas por capa"],
             ["Num layers",         "3",       "Capas del tronco compartido"]],
            col_widths=[4*cm, 2.5*cm, 9*cm],
        ),
        sp(8),
    ]),
]

# ── 5. Uso ────────────────────────────────────────────────────────────────────
story += [
    KeepTogether([
        h1("5. Uso"),
        hr(),
        p("<b>Arrancar PPO desde el mejor modelo REINFORCE</b> (recomendado):"),
        code("python train_ppo.py --load best_model.pt --episodes 5000"),
        p("<b>Continuar un checkpoint PPO existente:</b>"),
        code("python train_ppo.py --load model_ppo.pt --episodes 5000"),
        p("<b>Ver el progreso en tiempo real</b> (otra terminal):"),
        code("python plot.py --file metrics_ppo.csv"),
        sp(3),
        p("El entrenamiento guarda automáticamente:"),
        *bullets([
            "<font name='Courier'>best_model_ppo.pt</font> — mejor modelo según evaluación formal (cada 100 ep).",
            "<font name='Courier'>model_ppo.pt</font> — checkpoint periódico (cada 500 ep).",
            "<font name='Courier'>metrics_ppo.csv</font> — win rate por batch, eval formal, policy loss.",
        ]),
        sp(10),
    ]),
]

# ── 6. Próximo paso ───────────────────────────────────────────────────────────
story += [
    KeepTogether([
        h1("6. Próximo Paso — Self-Play"),
        hr(),
        p("Una vez estabilizado PPO vs heurístico, la Fase 4b incorporará <b>self-play</b>: "
          "el agente jugará contra versiones anteriores de sí mismo en lugar de (o además de) "
          "el heurístico fijo. Esto genera un currículo progresivo y evita el sobre-ajuste a "
          "una estrategia específica."),
        sp(3),
        tabla(
            [["Modalidad",         "Oponentes",                        "Ventaja"],
             ["PPO vs Heurístico", "3 × HeuristicPlayer",              "Baseline sólido, estable"],
             ["PPO vs REINFORCE",  "best_model.pt como oponente fijo", "Aprende a superar la Fase 3"],
             ["Self-play (league)","Pool de checkpoints históricos",   "Currículo automático, más robusto"]],
            col_widths=[4*cm, 6*cm, 5.5*cm],
        ),
        sp(14),
        HRFlowable(width="60%", thickness=0.5,
                   color=colors.HexColor("#aaaaaa"), hAlign="CENTER"),
        sp(6),
        Paragraph("Catan AI  ·  Fase 4  ·  Python + catanatron + PyTorch", CENTER),
    ]),
]

# ─── Generar ──────────────────────────────────────────────────────────────────

doc.build(story)
print("PDF generado: instrucciones/catan_ai_ppo.pdf")
