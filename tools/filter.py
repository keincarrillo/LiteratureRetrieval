import argparse
import csv
import html
import http.server
import json
import os
import re
import socket
import sys
import webbrowser

PALABRAS_POR_FRAGMENTO = 500
ID_INICIO = 1
META_DOCUMENTOS = 100

# Configuración por defecto (retrocompatibilidad: comportamiento original Kundera)
ARCHIVO_LIBRO = "La insoportable levedad del ser - Milan Kundera.txt"
ARCHIVO_SALIDA = "corpus_kundera.csv"
BOOK_ID = "kundera_levedad"
TITLE = "La insoportable levedad del ser"
AUTHOR = "Milan Kundera"
LANGUAGE = "es"
SOURCE = "Lectulandia"
URL = "https://ww3.lectulandia.com/book/la-insoportable-levedad-del-ser/"

MIN_PALABRAS_ACEPTABLE = int(PALABRAS_POR_FRAGMENTO * 0.75)

# ==============================================================================

MOTIVOS_KUNDERA = [
    r"\beterno retorno\b",
    r"\blevedad\b",
    r"\bpeso\b",
    r"\bkitsch\b",
    r"\bv[eé]rtigo\b",
    r"\bcompasi[oó]n\b",
    r"\bcasualidad\b",
    r"\bazar\b",
    r"\bmet[aá]fora\b",
    r"\btom[aá]s\b",
    r"\bteresa\b",
    r"\bsabina\b",
    r"\bfranz\b",
    r"\bkarenin\b",
    r"\bsombrero\b",
    r"\bhongo\b",
    r"\bcuerpo\b",
    r"\balma\b",
    r"\bespejo\b",
    r"\bdesnud[oa]s?\b",
    r"\bamante[s]?\b",
    r"\binfidelidad\b",
    r"\bcelos\b",
    r"\berotismo\b",
    r"\bpraga\b",
    r"\btanques?\b",
    r"\bruso[s]?\b",
    r"\bocupaci[oó]n\b",
    r"\bpolic[ií]a\b",
    r"\binterrogatori",
    r"\bbohemia\b",
    r"\bz[uú]rich\b",
    r"\bginebra\b",
    r"\bcampo\b",
]

CAPITULOS_PARRAFOS = {}
GLOBAL_CANDIDATOS = []
GLOBAL_MAP = {}


def cargar_config(ruta_config=None):
    global ARCHIVO_LIBRO, ARCHIVO_SALIDA, BOOK_ID, TITLE, AUTHOR, LANGUAGE, SOURCE, URL, META_DOCUMENTOS, MOTIVOS_KUNDERA
    if ruta_config is None:
        return
    try:
        with open(ruta_config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        print(f"[WARN] No se pudo cargar config {ruta_config}: {e}")
        return

    base_dir = os.path.dirname(os.path.abspath(ruta_config))
    ARCHIVO_LIBRO = cfg.get("txt") or ARCHIVO_LIBRO
    ARCHIVO_SALIDA = cfg.get("csv") or ARCHIVO_SALIDA
    BOOK_ID = cfg.get("book_id") or BOOK_ID
    TITLE = cfg.get("title") or TITLE
    AUTHOR = cfg.get("author") or AUTHOR
    LANGUAGE = cfg.get("language") or LANGUAGE
    SOURCE = cfg.get("source") or SOURCE
    URL = cfg.get("url") or URL
    if "meta_documentos" in cfg:
        try:
            META_DOCUMENTOS = int(cfg["meta_documentos"])
        except Exception:
            pass

    motivos = cfg.get("motivos_filter") or cfg.get("motivos") or []
    if motivos:
        MOTIVOS_KUNDERA = list(motivos)

    # Si las rutas son relativas, resuelve respecto a book_dir
    if not os.path.isabs(ARCHIVO_LIBRO):
        ARCHIVO_LIBRO = os.path.join(base_dir, ARCHIVO_LIBRO)
    if not os.path.isabs(ARCHIVO_SALIDA):
        ARCHIVO_SALIDA = os.path.join(base_dir, ARCHIVO_SALIDA)



def calcular_relevancia(texto):
    score = 0
    t_min = texto.lower()
    for patron in MOTIVOS_KUNDERA:
        score += len(re.findall(patron, t_min)) * 2

    dialogos = len(re.findall(r'(?:—|–|-|")[^—–\-"\n]{3,}', texto))
    score += min(dialogos, 8)
    return score


def generar_html_resaltado(texto):
    res = html.escape(texto)
    for p in MOTIVOS_KUNDERA:
        res = re.sub(
            f"({p})", r'<span class="hl">\1</span>', res, flags=re.IGNORECASE
        )
    return res


def limpiar_texto_crudo(texto):
    t = " ".join(texto.split()).strip()
    t = re.sub(r"^\s*\d+\s*[\.\-–—]?\s*", "", t)
    return t


def extraer_capitulos(ruta_archivo):
    with open(ruta_archivo, "r", encoding="utf-8") as f:
        contenido = f.read()

    bloques = re.split(r"^===\s*CAPÍTULO:\s*(.*?)\s*===", contenido, flags=re.M)
    capitulos = []
    parte_actual = "Inicio"

    for i in range(1, len(bloques), 2):
        encabezado = bloques[i].strip()
        cuerpo = bloques[i + 1].strip()

        if "parte" in encabezado.lower():
            parte_actual = encabezado
            continue

        if encabezado.lower() == TITLE.lower():
            continue

        etiqueta_cap = (
            f"{parte_actual} - Cap. {encabezado}"
            if parte_actual != "Inicio"
            else f"Cap. {encabezado}"
        )
        if len(cuerpo.split()) >= 60:
            capitulos.append((etiqueta_cap, cuerpo))

    return capitulos


def segmentar_libro(capitulos):
    candidatos = []
    uid = 0

    for etiqueta_cap, texto in capitulos:
        parrafos = [
            p.strip() for p in re.split(r"\n\s*\n+", texto) if p.strip()
        ]
        CAPITULOS_PARRAFOS[etiqueta_cap] = parrafos

        buffer_parrafos = []
        cuenta_palabras = 0
        idx_inicio = 0

        for i, parrafo in enumerate(parrafos):
            if not buffer_parrafos:
                idx_inicio = i

            palabras_parrafo = parrafo.split()
            buffer_parrafos.append(parrafo)
            cuenta_palabras += len(palabras_parrafo)

            if cuenta_palabras >= PALABRAS_POR_FRAGMENTO:
                frag_raw = limpiar_texto_crudo(" ".join(buffer_parrafos))
                uid += 1
                candidatos.append(
                    {
                        "uid": uid,
                        "capitulo": etiqueta_cap,
                        "p_inicio": idx_inicio,
                        "p_fin": i,
                        "total_parrafos": len(parrafos),
                        "texto_original": frag_raw,
                        "texto": frag_raw,
                        "html": generar_html_resaltado(frag_raw),
                        "palabras": len(frag_raw.split()),
                        "score": calcular_relevancia(frag_raw),
                        "accepted": False,
                    }
                )
                buffer_parrafos = []
                cuenta_palabras = 0

        if buffer_parrafos and cuenta_palabras >= MIN_PALABRAS_ACEPTABLE:
            frag_raw = limpiar_texto_crudo(" ".join(buffer_parrafos))
            uid += 1
            candidatos.append(
                {
                    "uid": uid,
                    "capitulo": etiqueta_cap,
                    "p_inicio": idx_inicio,
                    "p_fin": len(parrafos) - 1,
                    "total_parrafos": len(parrafos),
                    "texto_original": frag_raw,
                    "texto": frag_raw,
                    "html": generar_html_resaltado(frag_raw),
                    "palabras": len(frag_raw.split()),
                    "score": calcular_relevancia(frag_raw),
                    "accepted": False,
                }
            )

    return candidatos


def sincronizar_desde_csv():
    if not os.path.exists(ARCHIVO_SALIDA):
        return

    with open(ARCHIVO_SALIDA, "r", encoding="utf-8") as f:
        lector = list(csv.DictReader(f))

    for fila in lector:
        t_csv = fila.get("text", "").strip()
        cap_csv = fila.get("chapter/position", "").strip()
        if not t_csv:
            continue

        for c in GLOBAL_CANDIDATOS:
            if c["accepted"]:
                continue
            if c["capitulo"] == cap_csv and (
                t_csv in c["texto_original"] or c["texto_original"] in t_csv
            ):
                c["texto"] = t_csv
                c["palabras"] = len(t_csv.split())
                c["score"] = calcular_relevancia(t_csv)
                c["html"] = generar_html_resaltado(t_csv)
                c["accepted"] = True
                break
            elif t_csv == c["texto_original"]:
                c["accepted"] = True
                break


def guardar_csv_global():
    aceptados = [c for c in GLOBAL_CANDIDATOS if c["accepted"]]
    encabezados = [
        "fragment_id",
        "book_id",
        "title",
        "author",
        "chapter/position",
        "text",
        "language",
        "source",
        "url",
    ]

    filas = []
    for idx, item in enumerate(aceptados):
        filas.append(
            {
                "fragment_id": ID_INICIO + idx,
                "book_id": BOOK_ID,
                "title": TITLE,
                "author": AUTHOR,
                "chapter/position": item["capitulo"],
                "text": item["texto"],
                "language": LANGUAGE,
                "source": SOURCE,
                "url": URL,
            }
        )

    with open(ARCHIVO_SALIDA, "w", encoding="utf-8", newline="") as f:
        escritor = csv.DictWriter(
            f, fieldnames=encabezados, quoting=csv.QUOTE_MINIMAL
        )
        escritor.writeheader()
        escritor.writerows(filas)


HTML_PAGE = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Revisor & Editor | {TITLE}</title>
<style>
  :root {
    --bg: #f8fafc;
    --card: #ffffff;
    --border: #e2e8f0;
    --border-hover: #cbd5e1;
    --text: #0f172a;
    --text-muted: #64748b;
    --primary: #059669;
    --primary-hover: #047857;
    --accent: #0284c7;
    --hl-bg: #fef08a;
    --hl-text: #854d0e;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    padding-bottom: 60px;
    min-height: 100vh;
  }
  .container {
    width: 100%;
    max-width: 100%;
    padding: 12px 24px;
  }
  .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
    gap: 12px;
  }
  .title { font-size: 16px; font-weight: 700; color: #0f172a; }
  .subtitle { font-size: 12px; color: var(--text-muted); display: inline-flex; align-items: center; gap: 8px; }
  .progress-bar-bg { width: 100%; height: 4px; background: var(--border); border-radius: 2px; overflow: hidden; margin: 6px 0 12px; }
  .progress-bar-fill { height: 100%; background: var(--primary); transition: width 0.25s ease; }
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  }
  .card-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    align-items: center;
    margin-bottom: 10px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
  }
  .toolbar {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    align-items: center;
    margin-bottom: 12px;
    background: #f1f5f9;
    padding: 6px 10px;
    border-radius: 6px;
    border: 1px solid var(--border);
  }
  .badge {
    background: #f1f5f9;
    color: #334155;
    font-size: 11.5px;
    font-weight: 600;
    padding: 3px 8px;
    border-radius: 4px;
    border: 1px solid var(--border);
  }
  .badge-words { color: #0f172a; background: #e2e8f0; }
  .badge-score { color: #b45309; background: #fef3c7; border: 1px solid #fde68a; }
  .badge-status-accepted { color: #15803d; background: #dcfce7; border: 1px solid #bbf7d0; }
  .badge-status-pending { color: #64748b; background: #f8fafc; }
  .content {
    font-family: Charter, Georgia, "Times New Roman", serif;
    font-size: 15px;
    line-height: 1.55;
    color: #1e293b;
    user-select: text;
    text-align: justify;
  }
  .edit-textarea {
    width: 100%;
    min-height: 280px;
    background: #ffffff;
    color: #0f172a;
    border: 1px solid var(--accent);
    border-radius: 6px;
    padding: 12px;
    font-family: Charter, Georgia, serif;
    font-size: 15px;
    line-height: 1.55;
    outline: none;
  }
  .hl {
    background-color: var(--hl-bg);
    color: var(--hl-text);
    padding: 0 3px;
    border-radius: 3px;
    font-weight: 600;
  }
  .bottom-bar {
    position: fixed;
    bottom: 0; left: 0; right: 0;
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(8px);
    border-top: 1px solid var(--border);
    padding: 7px 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    z-index: 100;
    box-shadow: 0 -2px 6px rgba(0,0,0,0.03);
  }
  .btn {
    border: 1px solid var(--border);
    background: #ffffff;
    color: #334155;
    padding: 4px 10px;
    border-radius: 5px;
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 4px;
    transition: all 0.1s ease;
  }
  .btn:hover:not(:disabled) { background: #f8fafc; border-color: var(--border-hover); color: #0f172a; }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .btn-accent { background: #f0f9ff; border-color: #bae6fd; color: #0369a1; }
  .btn-accent:hover:not(:disabled) { background: #e0f2fe; }
  .btn-action { background: var(--primary); border-color: var(--primary); color: #ffffff; padding: 5px 14px; font-weight: 600; }
  .btn-action:hover { background: var(--primary-hover); }
  .btn-danger { background: #fee2e2; border-color: #fca5a5; color: #991b1b; padding: 5px 14px; font-weight: 600; }
  .btn-danger:hover { background: #fecaca; }
  .kbd {
    background: #f1f5f9;
    padding: 1px 4px;
    border-radius: 3px;
    font-size: 10px;
    border: 1px solid #cbd5e1;
    color: #475569;
    font-family: monospace;
  }
  .toast {
    position: fixed;
    top: 12px; right: 16px;
    background: #0f172a;
    color: #fff;
    padding: 6px 14px;
    border-radius: 6px;
    font-size: 12px;
    opacity: 0;
    transform: translateY(-4px);
    transition: all 0.2s ease;
    pointer-events: none;
    z-index: 200;
  }
  .toast.show { opacity: 1; transform: translateY(0); }
</style>
</head>
<body>
  <div id="toast" class="toast">Guardado</div>
  <div class="container">
    <div class="header">
      <div>
        <span class="title">{TITLE}</span>
        <span class="subtitle">{AUTHOR} &bull; <button id="btnSort" class="btn" style="padding: 2px 6px; font-size: 11px;">Relevancia ▼</button></span>
      </div>
      <div style="display: flex; gap: 12px; align-items: center;">
        <span style="font-size: 12px; color: var(--text-muted);" id="candIndexText">0 / 0</span>
        <span style="font-size: 13px; font-weight: 700; color: #059669;" id="progressText">0 / {META_DOCUMENTOS} aceptados</span>
      </div>
    </div>

    <div class="progress-bar-bg">
      <div class="progress-bar-fill" id="progressFill" style="width: 0%;"></div>
    </div>

    <div class="card">
      <div class="card-meta">
        <span class="badge" id="capBadge">Capítulo</span>
        <span class="badge badge-words" id="wordsBadge">0 palabras</span>
        <span class="badge badge-score" id="scoreBadge">0 pts</span>
        <span class="badge" id="statusBadge">Pendiente</span>
      </div>

      <div class="toolbar">
        <button class="btn btn-accent" id="btnAddPrev" title="Anexar párrafo anterior del capítulo">
          + Párrafo ant. <span class="kbd">[</span>
        </button>
        <button class="btn" id="btnDropFirst" title="Quitar primer párrafo">
          - 1er párr.
        </button>
        <button class="btn btn-accent" id="btnAddNext" title="Anexar párrafo siguiente del capítulo">
          + Párrafo sig. <span class="kbd">]</span>
        </button>
        <button class="btn" id="btnDropLast" title="Quitar último párrafo">
          - Últ. párr.
        </button>
        <span style="border-left: 1px solid var(--border); height: 16px; margin: 0 4px;"></span>
        <button class="btn" id="btnCropSelection" title="Subraya con el cursor y presiona aquí">
          ✂️ Recortar selección <span class="kbd">S</span>
        </button>
        <button class="btn" id="btnToggleEdit">
          ✏️ Edición manual
        </button>
        <button class="btn" id="btnRestoreOriginal">
          ↺ Restaurar
        </button>
      </div>

      <div class="content" id="fragmentText">Cargando...</div>
      <textarea class="edit-textarea" id="editTextarea" style="display: none;"></textarea>
    </div>
  </div>

  <div class="bottom-bar">
    <div style="display: flex; gap: 6px;">
      <button class="btn" id="btnBack">← Anterior <span class="kbd">B</span></button>
      <button class="btn" id="btnSkip">Omitir <span class="kbd">N</span></button>
    </div>
    <button class="btn btn-action" id="btnAccept">✓ Aceptar Fragmento <span class="kbd">Y</span></button>
  </div>

  <script>
    let candidates = [];
    let currentIndex = 0;
    let sortByScore = true;
    let isEditMode = false;

    async function loadData() {
      const res = await fetch('/api/data');
      const data = await res.json();
      candidates = data.candidates;
      aplicarOrden();

      const primerPendiente = candidates.findIndex(c => !c.accepted);
      currentIndex = primerPendiente !== -1 ? primerPendiente : 0;
      renderCurrent();
    }

    function aplicarOrden() {
      if (sortByScore) {
        candidates.sort((a, b) => b.score - a.score);
      } else {
        candidates.sort((a, b) => a.uid - b.uid);
      }
    }

    function renderCurrent() {
      if (!candidates.length) return;
      const c = candidates[currentIndex];

      document.getElementById('capBadge').innerText = c.capitulo;
      document.getElementById('wordsBadge').innerText = `${c.palabras} palabras`;
      document.getElementById('scoreBadge').innerText = `${c.score} pts`;

      const stBadge = document.getElementById('statusBadge');
      if (c.accepted) {
        stBadge.innerText = '✓ Guardado';
        stBadge.className = 'badge badge-status-accepted';
      } else {
        stBadge.innerText = 'Pendiente';
        stBadge.className = 'badge badge-status-pending';
      }

      document.getElementById('fragmentText').innerHTML = c.html;
      document.getElementById('editTextarea').value = c.texto;
      document.getElementById('candIndexText').innerText = `${currentIndex + 1} de ${candidates.length}`;

      const aceptados = candidates.filter(x => x.accepted).length;
      document.getElementById('progressText').innerText = `${aceptados} / {META_DOCUMENTOS} aceptados`;
      const pct = Math.min(100, Math.round((aceptados / {META_DOCUMENTOS}) * 100));
      document.getElementById('progressFill').style.width = `${pct}%`;

      document.getElementById('btnAddPrev').disabled = (c.p_inicio <= 0);
      document.getElementById('btnAddNext').disabled = (c.p_fin >= c.total_parrafos - 1);
      document.getElementById('btnDropFirst').disabled = (c.p_inicio >= c.p_fin);
      document.getElementById('btnDropLast').disabled = (c.p_fin <= c.p_inicio);

      const btnAcc = document.getElementById('btnAccept');
      if (c.accepted) {
        btnAcc.innerText = '✕ Quitar de CSV [Y]';
        btnAcc.className = 'btn btn-danger';
      } else {
        btnAcc.innerHTML = '✓ Aceptar Fragmento <span class="kbd">Y</span>';
        btnAcc.className = 'btn btn-action';
      }

      if (isEditMode) toggleEditMode(false);
    }

    function showToast(msg) {
      const t = document.getElementById('toast');
      t.innerText = msg;
      t.classList.add('show');
      setTimeout(() => t.classList.remove('show'), 1400);
    }

    function toggleEditMode(forceState) {
      isEditMode = forceState !== undefined ? forceState : !isEditMode;
      const textDiv = document.getElementById('fragmentText');
      const textarea = document.getElementById('editTextarea');
      const btn = document.getElementById('btnToggleEdit');

      if (isEditMode) {
        textDiv.style.display = 'none';
        textarea.style.display = 'block';
        textarea.focus();
        btn.innerText = '👁️ Vista lectura';
        btn.classList.add('btn-accent');
      } else {
        aplicarCambiosDeTexto(textarea.value);
        textDiv.style.display = 'block';
        textarea.style.display = 'none';
        btn.innerText = '✏️ Edición manual';
        btn.classList.remove('btn-accent');
      }
    }

    async function aplicarCambiosDeTexto(nuevoTexto) {
      const c = candidates[currentIndex];
      const limpio = nuevoTexto.trim();
      if (!limpio) return;

      c.texto = limpio;
      c.palabras = limpio.split(/\\s+/).filter(Boolean).length;

      const res = await fetch('/api/update_text', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ uid: c.uid, text: limpio })
      });
      const data = await res.json();
      c.html = data.html;
      c.score = data.score;
      renderCurrent();
    }

    async function shiftParagraph(action) {
      const c = candidates[currentIndex];
      const res = await fetch('/api/shift_paragraph', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ uid: c.uid, action })
      });
      const data = await res.json();
      if (data.status === 'ok') {
        c.texto = data.texto;
        c.html = data.html;
        c.palabras = data.palabras;
        c.score = data.score;
        c.p_inicio = data.p_inicio;
        c.p_fin = data.p_fin;
        renderCurrent();
      }
    }

    function recortarASeleccion() {
      const seleccion = window.getSelection().toString().trim();
      if (!seleccion) {
        showToast('Subraya texto primero para recortar');
        return;
      }
      aplicarCambiosDeTexto(seleccion);
      showToast('Recortado a selección');
    }

    function restaurarOriginal() {
      const c = candidates[currentIndex];
      aplicarCambiosDeTexto(c.texto_original);
      showToast('Texto original restaurado');
    }

    async function toggleAccept() {
      const c = candidates[currentIndex];

      if (isEditMode) {
        await aplicarCambiosDeTexto(document.getElementById('editTextarea').value);
        toggleEditMode(false);
      }

      const newState = !c.accepted;
      c.accepted = newState;

      await fetch('/api/toggle', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ uid: c.uid, accepted: newState, text: c.texto })
      });

      showToast(newState ? 'Guardado en CSV' : 'Eliminado de CSV');

      if (newState && currentIndex < candidates.length - 1) {
        currentIndex++;
      }
      renderCurrent();
    }

    function nextCandidate() {
      if (currentIndex < candidates.length - 1) {
        currentIndex++;
        renderCurrent();
      }
    }

    function prevCandidate() {
      if (currentIndex > 0) {
        currentIndex--;
        renderCurrent();
      }
    }

    document.getElementById('btnAccept').onclick = toggleAccept;
    document.getElementById('btnSkip').onclick = nextCandidate;
    document.getElementById('btnBack').onclick = prevCandidate;
    document.getElementById('btnCropSelection').onclick = recortarASeleccion;
    document.getElementById('btnToggleEdit').onclick = () => toggleEditMode();
    document.getElementById('btnRestoreOriginal').onclick = restaurarOriginal;

    document.getElementById('btnAddPrev').onclick = () => shiftParagraph('add_prev');
    document.getElementById('btnAddNext').onclick = () => shiftParagraph('add_next');
    document.getElementById('btnDropFirst').onclick = () => shiftParagraph('drop_first');
    document.getElementById('btnDropLast').onclick = () => shiftParagraph('drop_last');

    document.getElementById('editTextarea').addEventListener('input', (e) => {
      const words = e.target.value.trim().split(/\\s+/).filter(Boolean).length;
      document.getElementById('wordsBadge').innerText = `${words} palabras`;
    });

    document.getElementById('btnSort').onclick = () => {
      sortByScore = !sortByScore;
      document.getElementById('btnSort').innerText = sortByScore ? 'Relevancia ▼' : 'Posición ▼';
      aplicarOrden();
      currentIndex = 0;
      renderCurrent();
    };

    window.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') {
        if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
          e.preventDefault();
          toggleAccept();
        }
        return;
      }

      const key = e.key.toLowerCase();
      if (key === 'y' || key === 'enter') {
        e.preventDefault();
        toggleAccept();
      } else if (key === 's') {
        e.preventDefault();
        recortarASeleccion();
      } else if (key === '[') {
        e.preventDefault();
        shiftParagraph('add_prev');
      } else if (key === ']') {
        e.preventDefault();
        shiftParagraph('add_next');
      } else if (key === 'n' || key === ' ' || key === 'arrowright') {
        e.preventDefault();
        nextCandidate();
      } else if (key === 'b' || key === 'arrowleft') {
        e.preventDefault();
        prevCandidate();
      }
    });

    loadData();
  </script>
</body>
</html>
"""


def render_pagina():
    return (
        HTML_PAGE.replace("{TITLE}", html.escape(TITLE))
        .replace("{AUTHOR}", html.escape(AUTHOR))
        .replace("{META_DOCUMENTOS}", str(META_DOCUMENTOS))
    )


class Handler(http.server.BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        return

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(render_pagina().encode("utf-8"))
        elif self.path == "/api/data":
            self.send_response(200)
            self.send_header("Content-type", "application/json; charset=utf-8")
            self.end_headers()
            payload = {
                "candidates": GLOBAL_CANDIDATOS,
                "meta": META_DOCUMENTOS,
            }
            self.wfile.write(json.dumps(payload).encode("utf-8"))
        else:
            self.send_error(404)

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        datos = json.loads(body.decode("utf-8"))

        if self.path == "/api/toggle":
            uid = datos.get("uid")
            accepted = datos.get("accepted")
            texto_editado = datos.get("text")

            if uid in GLOBAL_MAP:
                c = GLOBAL_MAP[uid]
                c["accepted"] = accepted
                if texto_editado:
                    c["texto"] = texto_editado
                    c["palabras"] = len(texto_editado.split())
                    c["score"] = calcular_relevancia(texto_editado)
                    c["html"] = generar_html_resaltado(texto_editado)
                guardar_csv_global()

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode("utf-8"))

        elif self.path == "/api/update_text":
            uid = datos.get("uid")
            nuevo_texto = datos.get("text", "").strip()

            if uid in GLOBAL_MAP and nuevo_texto:
                c = GLOBAL_MAP[uid]
                c["texto"] = nuevo_texto
                c["palabras"] = len(nuevo_texto.split())
                c["score"] = calcular_relevancia(nuevo_texto)
                c["html"] = generar_html_resaltado(nuevo_texto)
                if c["accepted"]:
                    guardar_csv_global()

                resp = {
                    "status": "ok",
                    "html": c["html"],
                    "score": c["score"],
                    "palabras": c["palabras"],
                }
            else:
                resp = {"status": "error"}

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))

        elif self.path == "/api/shift_paragraph":
            uid = datos.get("uid")
            action = datos.get("action")

            if uid in GLOBAL_MAP and action:
                c = GLOBAL_MAP[uid]
                cap_parrafos = CAPITULOS_PARRAFOS.get(c["capitulo"], [])

                if action == "add_prev" and c["p_inicio"] > 0:
                    c["p_inicio"] -= 1
                    p_text = cap_parrafos[c["p_inicio"]]
                    c["texto"] = limpiar_texto_crudo(p_text + " " + c["texto"])
                elif (
                    action == "add_next"
                    and c["p_fin"] < len(cap_parrafos) - 1
                ):
                    c["p_fin"] += 1
                    p_text = cap_parrafos[c["p_fin"]]
                    c["texto"] = limpiar_texto_crudo(c["texto"] + " " + p_text)
                elif action == "drop_first" and c["p_inicio"] < c["p_fin"]:
                    c["p_inicio"] += 1
                    slice_p = cap_parrafos[c["p_inicio"] : c["p_fin"] + 1]
                    c["texto"] = limpiar_texto_crudo(" ".join(slice_p))
                elif action == "drop_last" and c["p_fin"] > c["p_inicio"]:
                    c["p_fin"] -= 1
                    slice_p = cap_parrafos[c["p_inicio"] : c["p_fin"] + 1]
                    c["texto"] = limpiar_texto_crudo(" ".join(slice_p))

                c["palabras"] = len(c["texto"].split())
                c["score"] = calcular_relevancia(c["texto"])
                c["html"] = generar_html_resaltado(c["texto"])

                if c["accepted"]:
                    guardar_csv_global()

                resp = {
                    "status": "ok",
                    "texto": c["texto"],
                    "html": c["html"],
                    "score": c["score"],
                    "palabras": c["palabras"],
                    "p_inicio": c["p_inicio"],
                    "p_fin": c["p_fin"],
                }
            else:
                resp = {"status": "error"}

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))
        else:
            self.send_error(404)


def encontrar_puerto_libre(puerto_inicial=8080):
    for p in range(puerto_inicial, puerto_inicial + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    return puerto_inicial


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Revisor y editor de fragmentos para corpus literario")
    parser.add_argument("--config", "-c", type=str, help="Ruta a book.json con metadatos y motivos")
    args = parser.parse_args()

    cargar_config(args.config)

    if not os.path.exists(ARCHIVO_LIBRO):
        print(f"Error: No se encontró '{ARCHIVO_LIBRO}'.")
        sys.exit(1)

    print("Indexando capítulos y segmentando fragmentos...")
    caps = extraer_capitulos(ARCHIVO_LIBRO)
    GLOBAL_CANDIDATOS = segmentar_libro(caps)
    GLOBAL_MAP = {c["uid"]: c for c in GLOBAL_CANDIDATOS}

    sincronizar_desde_csv()

    puerto = encontrar_puerto_libre(8080)
    servidor = http.server.HTTPServer(("127.0.0.1", puerto), Handler)

    url_local = f"http://127.0.0.1:{puerto}"
    print(f"\nServidor activo en: {url_local}")
    print(f"Destino CSV: {ARCHIVO_SALIDA}")
    print("Presiona Ctrl + C para detener el proceso.\n")

    webbrowser.open(url_local)

    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor cerrado. El CSV está guardado y actualizado.")
