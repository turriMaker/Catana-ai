"""
Genera catan_ai_doc.pdf usando reportlab.
Ejecutar una sola vez: python generate_pdf.py
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Preformatted, PageBreak, KeepTogether,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

# ─── Documento ────────────────────────────────────────────────────────────────

doc = SimpleDocTemplate(
    "instrucciones/catan_ai_doc.pdf",
    pagesize=A4,
    leftMargin=2.5*cm, rightMargin=2.5*cm,
    topMargin=2.5*cm, bottomMargin=2.5*cm,
    title="Catan AI — Documentación Técnica Fase 3",
    author="Catan AI Project",
)

W = A4[0] - 5*cm  # ancho útil

# ─── Estilos ──────────────────────────────────────────────────────────────────

base = getSampleStyleSheet()

def style(name, **kw):
    s = ParagraphStyle(name, **kw)
    return s

BODY    = style("body",    fontName="Helvetica",       fontSize=10, leading=15, spaceAfter=6,  alignment=TA_JUSTIFY)
H1      = style("h1",      fontName="Helvetica-Bold",  fontSize=16, leading=20, spaceBefore=18, spaceAfter=8, textColor=colors.HexColor("#1a3a6b"))
H2      = style("h2",      fontName="Helvetica-Bold",  fontSize=13, leading=17, spaceBefore=12, spaceAfter=6, textColor=colors.HexColor("#2c5f8a"))
H3      = style("h3",      fontName="Helvetica-BoldOblique", fontSize=11, leading=15, spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#444444"))
TITLE   = style("title",   fontName="Helvetica-Bold",  fontSize=28, leading=34, alignment=TA_CENTER, textColor=colors.HexColor("#1a3a6b"))
SUBT    = style("subt",    fontName="Helvetica",        fontSize=14, leading=18, alignment=TA_CENTER, textColor=colors.HexColor("#2c5f8a"), spaceAfter=4)
SMALL   = style("small",   fontName="Helvetica",        fontSize=8.5, leading=12, textColor=colors.HexColor("#555555"))
CENTER  = style("center",  fontName="Helvetica",        fontSize=10, leading=14, alignment=TA_CENTER)
BULLET  = style("bullet",  fontName="Helvetica",        fontSize=10, leading=14, leftIndent=14, spaceAfter=3, bulletIndent=4)
CODE_S  = style("code",    fontName="Courier",          fontSize=8.5, leading=12, leftIndent=10, backColor=colors.HexColor("#f5f5f5"), spaceAfter=6)

def code(text):
    return Preformatted(text.strip(), CODE_S, maxLineLength=90)

def h1(t):  return Paragraph(t, H1)
def h2(t):  return Paragraph(t, H2)
def h3(t):  return Paragraph(t, H3)
def p(t):   return Paragraph(t, BODY)
def sp(n=6): return Spacer(1, n)
def hr():   return HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc"), spaceAfter=8, spaceBefore=4)
def bullet(items):
    return [Paragraph(f"• {i}", BULLET) for i in items]

def tabla(data, col_widths=None, header=True):
    t = Table(data, colWidths=col_widths)
    cmds = [
        ("FONTNAME",  (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",  (0,0), (-1,-1), 9),
        ("BACKGROUND",(0,0), (-1,0),  colors.HexColor("#1a3a6b")),
        ("TEXTCOLOR", (0,0), (-1,0),  colors.white),
        ("ALIGN",     (0,0), (-1,-1), "CENTER"),
        ("ALIGN",     (0,1), (1,-1),  "LEFT"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#f0f4fa"), colors.white]),
        ("GRID",      (0,0), (-1,-1), 0.4, colors.HexColor("#bbbbbb")),
        ("TOPPADDING",(0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
    ]
    t.setStyle(TableStyle(cmds))
    return t

# ─── Contenido ────────────────────────────────────────────────────────────────

story = []

# ── Portada ──────────────────────────────────────────────────────────────────
story += [
    Spacer(1, 3*cm),
    Paragraph("Catan AI", TITLE),
    sp(8),
    Paragraph("Documentación Técnica — Fase 3", SUBT),
    Paragraph("Red Neuronal con Aprendizaje por Refuerzo", SUBT),
    sp(16),
    HRFlowable(width="60%", thickness=1, color=colors.HexColor("#1a3a6b"), hAlign="CENTER"),
    sp(16),
    tabla(
        [["Componente", "Detalle"],
         ["Entorno",   "catanatron"],
         ["Red neuronal", "PyTorch (CPU)"],
         ["Algoritmo", "REINFORCE"],
         ["Vector de estado", "250 features"],
         ["Espacio de acciones", "244 acciones (fijo + máscara)"]],
        col_widths=[7*cm, 8*cm],
    ),
    sp(24),
    HRFlowable(width="60%", thickness=0.5, color=colors.HexColor("#aaaaaa"), hAlign="CENTER"),
    sp(8),
    Paragraph("Python + catanatron + PyTorch + matplotlib", CENTER),
    PageBreak(),
]

# ── Sección 1: Descripción general ───────────────────────────────────────────
story += [
    h1("1. Descripción General"),
    hr(),
    p("Este documento describe la implementación de la <b>Fase 3</b> del proyecto Catan AI: "
      "el diseño, entrenamiento y evaluación de una red neuronal que aprende a jugar al Catan "
      "mediante aprendizaje por refuerzo (<i>Reinforcement Learning</i>, RL)."),
    sp(),
    h2("1.1 Benchmarks de referencia"),
    tabla(
        [["Agente", "Victorias / 200", "Tasa de victoria"],
         ["Aleatorio (Fase 1)", "~50", "~25%"],
         ["Heurístico (Fase 2)", "93", "46.5%"],
         ["Red neuronal (Fase 3)", "?", "objetivo: >46.5%"]],
        col_widths=[7*cm, 5*cm, 5*cm],
    ),
    sp(),
    h2("1.2 Estructura de archivos"),
    tabla(
        [["Archivo", "Propósito"],
         ["model.py", "Features, codificación de acciones, arquitectura de la red"],
         ["neural_player.py", "Jugador que usa la red para decidir"],
         ["train.py", "Bucle de entrenamiento REINFORCE"],
         ["eval.py", "Evaluación formal contra otros agentes"],
         ["plot.py", "Gráfico del progreso del entrenamiento"],
         ["heuristic_player.py", "Agente de Fase 2 (oponente de referencia)"]],
        col_widths=[5*cm, 11*cm],
    ),
]

# ── Sección 2: Estado ─────────────────────────────────────────────────────────
story += [
    sp(12),
    h1("2. Representación del Estado"),
    hr(),
    p("La red recibe el estado del juego como un vector numérico de dimensión <b>250</b>, "
      "construido en <font name='Courier'>model.py</font> por la función "
      "<font name='Courier'>extract_features(state, color)</font>."),
    sp(),
    h2("2.1 Composición del vector"),
    tabla(
        [["Índices", "Componente", "Dimensión"],
         ["0 – 99",   "Estado de los 4 jugadores (25 features × 4)", "100"],
         ["100 – 153","Edificios en los 54 nodos del tablero",         "54"],
         ["154 – 225","Caminos en las 72 aristas del tablero",         "72"],
         ["226 – 244","Posición del ladrón (one-hot, 19 fichas)",      "19"],
         ["245 – 249","Recursos del banco (normalizados)",             "5"],
         ["",          "Total",                                         "250"]],
        col_widths=[3.5*cm, 9*cm, 3*cm],
    ),
    sp(),
    h2("2.2 Features por jugador (25 por jugador)"),
    p("La perspectiva se <b>rota</b> de forma que el jugador RL aparece siempre en la posición P₀, "
      "permitiendo que la red aprenda de forma independiente del color asignado."),
    sp(4),
    *bullet([
        "<b>10 stats:</b> puntos de victoria (visibles y reales), caminos / colonias / ciudades "
        "disponibles, longitud del camino más largo, banderas de turno.",
        "<b>5 recursos en mano:</b> WOOD, BRICK, SHEEP, WHEAT, ORE.",
        "<b>5 cartas de desarrollo en mano:</b> KNIGHT, YEAR_OF_PLENTY, MONOPOLY, "
        "ROAD_BUILDING, VICTORY_POINT.",
        "<b>5 cartas de desarrollo jugadas.</b>",
    ]),
    sp(),
    h2("2.3 Codificación del tablero"),
    *bullet([
        "<b>Nodos (54):</b> +1 colonia propia, +2 ciudad propia, −1 colonia rival, −2 ciudad rival, 0 vacío.",
        "<b>Aristas (72):</b> +1 camino propio, −1 camino rival, 0 sin camino.",
        "<b>Ladrón:</b> vector one-hot de 19 posiciones (una por ficha de tierra).",
        "<b>Banco:</b> cada recurso normalizado dividiendo por 19 (cantidad inicial máxima).",
    ]),
]

# ── Sección 3: Espacio de acciones ────────────────────────────────────────────
story += [
    sp(12),
    h1("3. Espacio de Acciones"),
    hr(),
    p("Se define un espacio de acciones <b>fijo de dimensión 244</b>, con una máscara binaria "
      "que anula las acciones inválidas en cada estado antes de aplicar softmax."),
    sp(),
    tabla(
        [["Índices", "Tipo de acción", "Cantidad"],
         ["0",        "ROLL",                               "1"],
         ["1",        "END_TURN",                           "1"],
         ["2",        "BUY_DEVELOPMENT_CARD",               "1"],
         ["3",        "PLAY_KNIGHT_CARD",                   "1"],
         ["4",        "PLAY_ROAD_BUILDING",                 "1"],
         ["5 – 58",   "BUILD_SETTLEMENT (nodo 0–53)",       "54"],
         ["59 – 112", "BUILD_CITY (nodo 0–53)",             "54"],
         ["113 – 184","BUILD_ROAD (arista 0–71)",           "72"],
         ["185 – 203","MOVE_ROBBER (ficha 0–18)",           "19"],
         ["204 – 208","PLAY_MONOPOLY (recurso 0–4)",        "5"],
         ["209 – 223","PLAY_YEAR_OF_PLENTY (15 combos)",    "15"],
         ["224 – 243","MARITIME_TRADE (20 combos)",         "20"],
         ["",         "Total",                              "244"]],
        col_widths=[3.5*cm, 9*cm, 3*cm],
    ),
    sp(4),
    p("La acción <font name='Courier'>DISCARD</font> no se incluye en el espacio fijo por su "
      "complejidad combinatoria; se maneja con selección aleatoria entre las opciones válidas."),
]

# ── Sección 4: Arquitectura ───────────────────────────────────────────────────
story += [
    sp(12),
    h1("4. Arquitectura de la Red Neuronal"),
    hr(),
    p("La red implementada en <font name='Courier'>CatanNet</font> es un perceptrón multicapa "
      "(<i>Multi-Layer Perceptron</i>, MLP):"),
    sp(4),
    Paragraph("<b>entrada (250)  →  [Linear(256) → ReLU] × 3  →  salida (244)</b>", CENTER),
    sp(8),
    code("""\
class CatanNet(nn.Module):
    def __init__(self, hidden_size=256, num_layers=3):
        super().__init__()
        layers = [nn.Linear(STATE_SIZE, hidden_size), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers.append(nn.Linear(hidden_size, ACTION_SPACE_SIZE))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # retorna logits (244,)"""),
    sp(4),
    p("La salida son <b>logits</b> (sin softmax). La selección de acción aplica:"),
    *bullet([
        "Mascarar logits inválidos con −∞.",
        "Aplicar softmax → distribución de probabilidad sobre acciones válidas.",
        "<b>Entrenamiento:</b> muestrear de la distribución (exploración estocástica).",
        "<b>Evaluación:</b> argmax (acción más probable).",
    ]),
]

# ── Sección 5: REINFORCE ──────────────────────────────────────────────────────
story += [
    sp(12),
    h1("5. Algoritmo de Entrenamiento: REINFORCE"),
    hr(),
    h2("5.1 Fundamento teórico"),
    p("El objetivo es maximizar la recompensa esperada J(θ) = E[R]. "
      "El gradiente de la política es:"),
    sp(4),
    Paragraph(
        "∇J(θ)  =  E[ Σₜ  ∇ log π(aₜ | sₜ)  ·  R ]",
        style("math", fontName="Courier-Bold", fontSize=11, alignment=TA_CENTER, spaceAfter=8),
    ),
    p("donde R ∈ {+1, −1} es la recompensa al final del juego (victoria o derrota)."),
    sp(),
    h2("5.2 Función de pérdida"),
    Paragraph(
        "L(θ)  =  −(1/T) · Σₜ  log π(aₜ | sₜ)  ·  R",
        style("math2", fontName="Courier-Bold", fontSize=11, alignment=TA_CENTER, spaceAfter=8),
    ),
    sp(),
    h2("5.3 Hiperparámetros"),
    tabla(
        [["Parámetro", "Valor"],
         ["Optimizador", "Adam"],
         ["Learning rate", "1 × 10⁻⁴"],
         ["Capas ocultas", "3 × 256 neuronas"],
         ["Recompensa", "+1 victoria, −1 derrota"],
         ["Clipping de gradiente", "‖∇‖ ≤ 1.0"]],
        col_widths=[7*cm, 9*cm],
    ),
    sp(),
    h2("5.4 Bucle de entrenamiento"),
    code("""\
def run_episode(model, optimizer):
    rl_player = NeuralPlayer(Color.RED, model, training=True)
    others    = [HeuristicPlayer(c) for c in [BLUE, WHITE, ORANGE]]
    game = Game([rl_player] + others)
    game.play()

    reward    = +1.0 if game.winning_color() == Color.RED else -1.0
    log_probs = torch.stack(rl_player.log_probs)
    loss      = -(log_probs * reward).mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()"""),
]

# ── Sección 6: Uso ────────────────────────────────────────────────────────────
story += [
    sp(12),
    h1("6. Guía de Uso"),
    hr(),
    h2("6.1 Activar entorno"),
    code("""\
cd C:\\Users\\franc\\Documents\\catan-ai
venv\\Scripts\\activate"""),
    sp(),
    h2("6.2 Entrenamiento"),
    code("""\
# Entrenamiento básico (5000 episodios vs heurístico)
python train.py

# Empezar vs aleatorio (más rápido para las primeras épocas)
python train.py --episodes 2000 --opponents random --save model_pre.pt

# Continuar desde un checkpoint
python train.py --load model_pre.pt --episodes 5000"""),
    sp(4),
    p("El entrenamiento guarda automáticamente:"),
    *bullet([
        "<font name='Courier'>best_model.pt</font> — mejor modelo según evaluación formal.",
        "<font name='Courier'>model.pt</font> — checkpoint cada 500 episodios.",
        "<font name='Courier'>metrics.csv</font> — historial de tasas de victoria.",
    ]),
    sp(),
    h2("6.3 Evaluación"),
    code("""\
# Evaluar el mejor modelo (200 partidas)
python eval.py

# Especificar checkpoint y número de partidas
python eval.py --model model.pt --games 400"""),
    sp(),
    h2("6.4 Visualización"),
    code("""\
# Abrir gráfico interactivo
python plot.py

# Guardar como imagen
python plot.py --save grafico.png"""),
    sp(4),
    p("El gráfico muestra: win rate durante entrenamiento (línea azul suavizada), "
      "evaluaciones formales cada 100 episodios (puntos rojos), baseline del 25% "
      "(línea gris) y objetivo del 46.5% (línea naranja)."),
]

# ── Sección 7: Flujo recomendado ──────────────────────────────────────────────
story += [
    sp(12),
    h1("7. Flujo Completo Recomendado"),
    hr(),
    p("<b>Paso 1</b> — Pre-entrenamiento rápido contra jugadores aleatorios:"),
    code("python train.py --episodes 2000 --opponents random --save model_pre.pt"),
    p("<b>Paso 2</b> — Entrenamiento principal contra el heurístico:"),
    code("python train.py --load model_pre.pt --episodes 8000 --save model_final.pt"),
    p("<b>Paso 3</b> — Monitorear el progreso (en otra terminal):"),
    code("python plot.py"),
    p("<b>Paso 4</b> — Evaluación final:"),
    code("python eval.py --model best_model.pt --games 200"),
]

# ── Sección 8: Desafíos ───────────────────────────────────────────────────────
story += [
    sp(12),
    h1("8. Desafíos Técnicos"),
    hr(),
    KeepTogether([
        h3("Credit assignment"),
        p("La recompensa llega al final del juego (50–100 decisiones). REINFORCE atribuye el "
          "mismo peso a todas las decisiones, lo cual es una aproximación gruesa. La Fase 4 "
          "aborda esto con PPO y descuento de recompensas."),
    ]),
    KeepTogether([
        h3("Información imperfecta"),
        p("El estado actual incluye las cartas de todos los jugadores (información completa). "
          "Una mejora futura es enmascarar los recursos de los rivales para simular el juego real."),
    ]),
    KeepTogether([
        h3("Acción DISCARD"),
        p("Cuando un jugador tiene más de 7 cartas al caer un 7, debe descartar la mitad. "
          "Esta acción tiene un espacio combinatorio enorme y se maneja con selección aleatoria "
          "en la implementación actual."),
    ]),
    KeepTogether([
        h3("Múltiples jugadores"),
        p("REINFORCE estándar asume un entorno de un solo agente. Con 4 jugadores el entorno "
          "es no-estacionario desde la perspectiva de la red."),
    ]),
]

# ── Sección 9: Próximos pasos ─────────────────────────────────────────────────
story += [
    sp(12),
    h1("9. Próximos Pasos — Fase 4"),
    hr(),
    *bullet([
        "Implementar <b>PPO</b> (Proximal Policy Optimization) para entrenamiento más estable.",
        "<b>Self-play</b> contra versiones anteriores de la red en lugar del heurístico fijo.",
        "Añadir <b>función de valor</b> (actor-critic) para reducir la varianza del gradiente.",
        "Modelar el <b>comercio marítimo</b> de forma más completa en el espacio de acciones.",
        "Explorar <b>comercio entre jugadores</b> (el mayor desafío único de Catan).",
    ]),
    sp(24),
    HRFlowable(width="60%", thickness=0.5, color=colors.HexColor("#aaaaaa"), hAlign="CENTER"),
    sp(8),
    Paragraph("Proyecto Catan AI  ·  Fase 3  ·  Python + catanatron + PyTorch", CENTER),
]

# ─── Generar ──────────────────────────────────────────────────────────────────

doc.build(story)
print("PDF generado: instrucciones/catan_ai_doc.pdf")
