#!/usr/bin/env python3
import sys
import warnings
from pathlib import Path
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

# Silenciar advertencias internas de ebooklib
warnings.filterwarnings("ignore")

BANDERA = "=== CAPÍTULO: {titulo} ==="

# Palabras clave en el nombre del archivo para descartar metadatos y créditos
IGNORAR = {
    "sinopsis", "info", "autor", "portada", "cover", 
    "titlepage", "creditos", "notas", "ded01", "ded02"
}

def convertir_epub(ruta_epub: Path, ruta_txt: Path):
    try:
        libro = epub.read_epub(str(ruta_epub))
    except Exception as e:
        print(f"  [ERROR] No se pudo leer {ruta_epub.name}: {e}")
        return

    # 1. Mapear nombres de archivo a títulos desde el TOC (índice oficial)
    mapa_titulos = {}

    def parse_toc(items):
        for item in items:
            if isinstance(item, tuple):
                seccion, subitems = item
                if hasattr(seccion, 'href') and hasattr(seccion, 'title'):
                    mapa_titulos[seccion.href.split('#')[0]] = seccion.title.strip()
                parse_toc(subitems)
            elif isinstance(item, epub.Link):
                mapa_titulos[item.href.split('#')[0]] = item.title.strip()

    parse_toc(libro.toc)

    # 2. Iterar en el orden secuencial del spine
    with open(ruta_txt, "w", encoding="utf-8") as salida:
        for spine_item in libro.spine:
            item_id, _ = spine_item
            item = libro.get_item_with_id(item_id)

            if not item or item.get_type() != ebooklib.ITEM_DOCUMENT:
                continue

            nombre_base = Path(item.get_name()).stem.lower()

            # Descartar archivos que son solo metadatos o portadas
            if any(ign in nombre_base for ign in IGNORAR):
                continue

            soup = BeautifulSoup(item.get_content(), "html.parser")

            # Eliminar estilos, scripts e imágenes
            for tag in soup(["style", "script", "img", "svg"]):
                tag.decompose()

            # Extraer párrafos limpios
            parrafos = [
                p.get_text().strip()
                for p in soup.find_all(["p", "h1", "h2", "h3", "h4"])
                if p.get_text().strip()
            ]
            cuerpo = "\n\n".join(parrafos)

            if not cuerpo:
                continue

            # Buscar título formal en TOC o en cabeceras h1/h2/h3
            titulo = mapa_titulos.get(item.get_name())
            if not titulo:
                primera_cabecera = soup.find(["h1", "h2", "h3"])
                if primera_cabecera and primera_cabecera.get_text().strip():
                    titulo = primera_cabecera.get_text().strip()

            # Si es un archivo secundario sin título propio (e.g. 'primer2', 'segundo2'),
            # se anexa el texto sin generar una bandera nueva
            if not titulo and nombre_base[-1].isdigit():
                salida.write(f"\n\n{cuerpo}\n")
            else:
                titulo_final = titulo if titulo else Path(item.get_name()).stem
                salida.write(f"\n\n{BANDERA.format(titulo=titulo_final)}\n\n")
                salida.write(cuerpo)
                salida.write("\n")

def main():
    directorio = Path(".") if len(sys.argv) < 2 else Path(sys.argv[1])
    archivos_epub = list(directorio.glob("*.epub"))

    if not archivos_epub:
        print("No se encontraron archivos .epub en el directorio.")
        return

    print(f"Procesando {len(archivos_epub)} libro(s)...")
    for epub_file in archivos_epub:
        txt_file = epub_file.with_suffix(".txt")
        print(f"-> Convirtiendo: {epub_file.name}")
        convertir_epub(epub_file, txt_file)
    print("¡Listo! Todos los archivos fueron convertidos.")

if __name__ == "__main__":
    main()
