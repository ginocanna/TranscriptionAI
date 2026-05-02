import streamlit as st
from google import genai
import subprocess
import os
import shutil
import time
import tempfile
import re
import json
from dotenv import load_dotenv
from pypdf import PdfReader
import PIL.Image

# Caricamento delle variabili d'ambiente (per GEMINI_API_KEY)
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=env_path, override=True)

# Configurazione della pagina
st.set_page_config(page_title="Appunti Accademici AI", page_icon="🎓", layout="wide")

# CSS Personalizzato per un look Premium
st.markdown("""
<style>
    /* Styling bottoni premium */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    /* Styling per testo e headers */
    h1 {
        font-weight: 800 !important;
        letter-spacing: -1.5px !important;
        color: #111827 !important;
        margin-bottom: 0 !important;
    }
    h2, h3 {
        font-weight: 700 !important;
        letter-spacing: -0.5px !important;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #4b5563;
        font-weight: 300;
        margin-bottom: 2rem;
    }
    /* Estetica dei container (es. le tab) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: transparent;
        font-weight: 600;
        color: #4F46E5 !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🎓 Appunti Accademici AI</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Trascrivi un audio con Whisper, miglioralo con Google Gemini e mantieni glossari personalizzati per materia. Eseguito in locale per la tua privacy.</div>", unsafe_allow_html=True)

# DB Materie
MATERIE_FILE = "materie.json"

def load_materie():
    if os.path.exists(MATERIE_FILE):
        with open(MATERIE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_materie(materie):
    with open(MATERIE_FILE, "w", encoding="utf-8") as f:
        json.dump(materie, f, indent=4, ensure_ascii=False)

materie_db = load_materie()

# --- SIDEBAR CONFIGURAZIONE ---
with st.sidebar:
    st.header("Configurazione Globale")
    
    # Inserimento opzionale o caricamento della chiave
    api_key_input = st.text_input("Gemini API Key (opzionale se nel .env)", type="password", value=os.getenv("GEMINI_API_KEY", ""))
    
    st.header("Impostazioni Base")
    asr_model_size = st.selectbox("Grandezza Modello Whisper", ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"], index=2, help="Modelli più grandi sono più accurati ma richiedono più RAM e sono più lenti.")
    lingua_audio = st.selectbox("Lingua Audio", ["it", "en", "auto"], index=2, help="Specificare la lingua evita traduzioni involontarie e aumenta la precisione.")
    gemini_model = st.selectbox("Modello Gemini", ["gemini-3.1-pro-preview", "gemini-3-flash-preview", "gemini-3.1-flash-lite", "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"], index=1)
    
# Layout a TAB
tab_trascrizione, tab_studio, tab_chat, tab_archivio, tab_materie = st.tabs([
    "📝 Trascrizione Lezione", 
    "🎓 Strumenti di Studio",
    "💬 Chat con gli Appunti",
    "🗂 Archivio Lezioni",
    "📚 Materie e Glossari"
])


# Espressioni Regolari per effettuare l'escape in LaTeX (molto basico, estendibile)
def escape_latex(text):
    # L'escape distruttivo rompeva la matematica (es. \$, \_, \^).
    # Ora chiediamo a Gemini di generare codice LaTeX valido direttamente.
    return text

@st.cache_resource(show_spinner=False)
def get_asr_model(size="small"):
    from faster_whisper import WhisperModel
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if torch.cuda.is_available() else "int8"
    try:
        model = WhisperModel(size, device=device, compute_type=compute_type)
        return model
    except Exception as e:
        st.error(f"Errore nel caricamento del modello {size}: {e}")
        return None

with tab_materie:
    st.header("📚 Gestione Materie e Glossari")
    st.write("Crea materie e carica slide o dispense PDF per estrarre automaticamente un glossario specifico. Questo aiuterà Whisper e Gemini a non sbagliare i termini tecnici e migliorerà enormemente la trascrizione.")
    
    col_mat1, col_mat2 = st.columns([1, 2])
    
    with col_mat1:
        st.subheader("Le tue Materie")
        nuova_materia = st.text_input("Aggiungi nuova materia", placeholder="Es. Fisica Tecnica")
        if st.button("Crea Materia"):
            if nuova_materia and nuova_materia not in materie_db:
                materie_db[nuova_materia] = ""
                save_materie(materie_db)
                st.success(f"Materia '{nuova_materia}' creata!")
                st.rerun()
                
        if not materie_db:
            st.info("Nessuna materia creata.")
            materia_selezionata = None
        else:
            materia_selezionata = st.radio("Seleziona materia per modificarla:", list(materie_db.keys()))
            if st.button("Elimina Materia", type="primary"):
                del materie_db[materia_selezionata]
                save_materie(materie_db)
                st.rerun()

    with col_mat2:
        if materia_selezionata:
            st.subheader(f"Glossario per: {materia_selezionata}")
            
            glossario_attuale = st.text_area("Glossario / Nomi (modificabile a mano)", value=materie_db[materia_selezionata], height=150, help="Questi termini verranno inviati a Whisper come prompt iniziale e a Gemini come istruzioni di revisione.")
            if st.button("Salva Glossario Modificato"):
                materie_db[materia_selezionata] = glossario_attuale
                save_materie(materie_db)
                st.success("Salvato con successo.")
            
            st.divider()
            st.subheader("Autocompleta da PDF (Slide/Dispense)")
            pdf_docs = st.file_uploader("Carica uno o più PDF della materia per estrarre i termini extra", type=["pdf"], accept_multiple_files=True)
            if st.button("Estrai Termini Tecnici"):
                if not api_key_input:
                    st.error("Inserisci la API Key di Gemini nella barra laterale o nel file .env prima di poter estrarre i termini dai PDF.")
                elif not pdf_docs:
                    st.warning("Carica almeno un PDF.")
                else:
                    with st.spinner("Estrazione testo e generazione glossario in corso..."):
                        client = genai.Client(api_key=api_key_input)
                        try:
                            testo_totale = ""
                            for pdf in pdf_docs:
                                reader = PdfReader(pdf)
                                for page in reader.pages:
                                    page_text = page.extract_text()
                                    if page_text:
                                        testo_totale += page_text + "\n"
                            
                            # Prevenzione OOM e limiti enormi token prompt
                            if len(testo_totale) > 200000:
                                testo_totale = testo_totale[:200000]
                                
                            prompt_estrazione = f"""
                            Analizza il seguente testo tratto da dispense/slide universitarie ed estrai un elenco compatto (separato da virgole)
                            dei **termini tecnici più importanti, nomi propri, acronimi e formule specifiche** della materia in questione.
                            L'obiettivo è fornire questo elenco come "glossario" a un sistema di trascrizione audio (Whisper) per non fargli sbagliare i termini specifici.
                            NON spiegare i termini, elencali soltanto, il più preciso possibile.
                            
                            Testo:
                            {testo_totale}
                            """
                            response = client.models.generate_content(
                                model=gemini_model,
                                contents=prompt_estrazione
                            )
                            nuovi_termini = response.text
                            
                            vecchio = materie_db[materia_selezionata]
                            nuovo_glossario = vecchio + (", " if vecchio else "") + nuovi_termini.strip()
                            materie_db[materia_selezionata] = nuovo_glossario
                            save_materie(materie_db)
                            
                            st.success("Termini estratti e aggiunti al glossario! Ricarico...")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Errore durante l'estrazione: {e}")

with tab_trascrizione:
    st.header("📝 Avvia Trascrizione e Miglioramento")
    
    col_t1, col_t2 = st.columns([1, 2])
    with col_t1:
        st.subheader("Contesto Lezione")
        if not materie_db:
            st.info("Nessuna materia presente. Vai nella tab 'Gestione Materie e Glossari' per crearne una ed avere un glossario associato.")
            materia_trascrizione = "Nessuna"
        else:
            opzioni_materie = ["Nessuna"] + list(materie_db.keys())
            materia_trascrizione = st.selectbox("Materia (usa il glossario)", opzioni_materie, help="Selezionando una materia, il suo glossario verrà fornito a Whisper e a Gemini per una precisione estrema.")
            
        custom_instructions = st.text_area(
            "Istruzioni Extra per Gemini", 
            placeholder="Es: formatta tutto come un riassunto a punti, oppure usa uno stile molto accademico...",
            help="Ulteriori direttive specifiche per il modello AI in fase di formattazione LaTeX."
        )
        
        import datetime
        data_lezione = st.date_input("Data della Lezione", value=datetime.date.today())

    with col_t2:
        st.subheader("Caricamento Audio e Materiale Extra")
        uploaded_audio = st.file_uploader("Lezione Audio (es. MP3, WAV, M4A)", type=["mp3", "wav", "m4a", "ogg"])
        uploaded_extras = st.file_uploader("Allegati Extra (PDF, PNG, JPG) per migliorare formule/lessico", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True)
        
        if uploaded_audio is not None:
            if st.button("Avvia Elaborazione Pipeline", type="primary", use_container_width=True):
                if not api_key_input:
                    st.error("Inserisci la Gemini API Key nella sidebar o nel file .env!")
                    st.stop()
                    
                client = genai.Client(api_key=api_key_input)
                
                with st.status("Inizializzazione...", expanded=True) as status:
                    try:
                        status.update(label="1/4. Trascrizione in corso con Whisper... (potrebbe richiedere tempo per file lunghi)", state="running")
                        
                        inc_glossary = ""
                        if materia_trascrizione != "Nessuna" and materie_db.get(materia_trascrizione):
                            inc_glossary = materie_db[materia_trascrizione]
                            
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio:
                            tmp_audio.write(uploaded_audio.read())
                            tmp_audio_path = tmp_audio.name
                        
                        model = get_asr_model(asr_model_size)
                        
                        options = {}
                        if lingua_audio != "auto":
                            options["language"] = lingua_audio
                        if inc_glossary.strip():
                            options["initial_prompt"] = inc_glossary.strip()
                            
                        try:
                            segments, info = model.transcribe(tmp_audio_path, **options)
                            total_duration = info.duration
                            
                            progress_bar = st.progress(0.0, text="Avvio trascrizione...")
                            
                            testo_grezzo_parts = []
                            for segment in segments:
                                testo_grezzo_parts.append(segment.text)
                                percent = min(1.0, segment.end / total_duration)
                                progress_bar.progress(percent, text=f"Trascrizione in corso: {int(percent * 100)}% ({segment.end:.1f}s / {total_duration:.1f}s)")
                                
                            testo_grezzo = " ".join(testo_grezzo_parts)
                            progress_bar.empty()
                        except Exception as e:
                            st.warning(f"Metodo di trascrizione progressivo fallito, provo fallback. Dettagli: {e}")
                            result = model.transcribe(tmp_audio_path, **options)
                            if isinstance(result, tuple) and len(result) == 2:
                                segments, _ = result
                                testo_grezzo = " ".join([segment.text for segment in segments])
                            else:
                                testo_grezzo = result.get("text", str(result))
                        
                        os.remove(tmp_audio_path)
                        
                        with st.expander("Mostra Testo Grezzo Whisper"):
                            st.text(testo_grezzo)
                            
                        status.update(label=f"2/4. Miglioramento intelligente con {gemini_model}...", state="running")
                        
                        base_prompt = f"""Agisci come un revisore accademico esperto e un trascrittore professionale.
Usa il seguente testo grezzo, trascritto automaticamente da una lezione universitaria (materia: {materia_trascrizione}), e miglioralo applicando le seguenti regole rigorose:
1. Correggi TUTTI gli errori di ortografia, punteggiatura, sintassi.
2. Elimina false partenze, balbettii e riempitivi tipici del parlato ("eh", "cioè").
3. Deduci e correggi i termini tecnici o accademici che il sistema speech-to-text potrebbe aver frainteso.
{"!! IMPORTANTE: Usa OBBLIGATORIAMENTE questo GLOSSARIO DELLA MATERIA per risolvere ambiguita' fonetiche: " + inc_glossary if inc_glossary else ""}
4. Dividi il testo in paragrafi logici ben strutturati per migliorarne la leggibilità.
5. Inserisci punti elenco se il professore elenca degli elementi.
6. Evidenzia in grassetto (es. **concetto limitato**) le definizioni principali. Usa le formule in notazione matematica LaTeX se appropriato.
7. ASSOLUTAMENTE CRITICO: Le parole del professore NON devono essere assolutamente riassunte o tagliate. Devono solamente essere corrette (sintassi, punteggiatura) e, in caso, implementate e integrate da formule in LaTeX, diagrammi o concetti estratti direttamente dai PDF/immagini allegati alla lezione.
8. ASSOLUTAMENTE CRITICO: NON tradurre il testo per nessun motivo. Mantieni la LINGUA ORIGINALE in cui è stata trascritta la lezione.
"""
                        if uploaded_extras:
                           base_prompt += "\n\nATTENZIONE: Ti ho fornito delle immagini / documenti allegati a questa trascrizione. Usali per capire meglio le formule matematiche e le scritte alla lavagna e integrarle nel markup LaTeX del risultato! Migliora i nomi, diagrammi o formule che vedi nel documento al fine di correggere i fraintendimenti della trascrizione.\n"

                        if custom_instructions.strip():
                            base_prompt += f"\n\nIstruzioni aggiuntive dell'utente:\n{custom_instructions.strip()}\n"

                        prompt = f"{base_prompt}\nTesto originale:\n{testo_grezzo}\n"
                        
                        content_parts = [prompt]
                        
                        if uploaded_extras:
                            for extra_file in uploaded_extras:
                                file_ext = extra_file.name.lower().split('.')[-1]
                                if file_ext in ['png', 'jpg', 'jpeg']:
                                    img = PIL.Image.open(extra_file)
                                    content_parts.append(img)
                                elif file_ext == 'pdf':
                                    try:
                                        reader = PdfReader(extra_file)
                                        pdf_text = ""
                                        for page in reader.pages:
                                            pdf_text += page.extract_text() + "\n"
                                        content_parts.append(f"\nContenuto del PDF {extra_file.name}:\n{pdf_text}\n")
                                    except Exception as e:
                                        print(f"Errore lettura PDF: {e}")

                        try:
                            response = client.models.generate_content(
                                model=gemini_model,
                                contents=content_parts
                            )
                            testo_migliorato = response.text
                        except Exception as e_gem:
                            st.warning(f"Errore generazione Gemini: {e_gem}\nProcedo con il salvataggio del solo testo grezzo.")
                            testo_migliorato = f"*[Nessun miglioramento disponibile a causa di un errore API, ti prego di ritentare. Testo trascritto grezzo di seguito]*\n\n{testo_grezzo}"

                        with st.expander("Mostra Testo Migliorato Gemini"):
                            st.markdown(testo_migliorato)
                            
                        status.update(label="3/4. Generazione codice LaTeX...", state="running")
                        
                        testo_latex_safe = escape_latex(testo_migliorato)
                        titolo_doc = "Appunti della Lezione"
                        if materia_trascrizione != "Nessuna":
                            titolo_doc += f" di {escape_latex(materia_trascrizione)}"
                        titolo_doc += f" del {data_lezione.strftime('%d/%m/%Y')}"
                        
                        safe_mat_filename = materia_trascrizione.replace(' ', '_').replace('/', '_')
                        data_str = data_lezione.strftime('%Y-%m-%d')
                        if materia_trascrizione == "Nessuna":
                            final_name = f"Appunti_{data_str}"
                        else:
                            final_name = f"{safe_mat_filename}_{data_str}"
                            
                        latex_template = fr"""\documentclass{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage{{geometry}}
\usepackage{{hyperref}}
\geometry{{a4paper, margin=1in}}

\title{{{titolo_doc}}}
\author{{Generato da AI Pipeline}}
\date{{{data_lezione.strftime('%d/%m/%Y')}}}

\begin{{document}}
\maketitle

{testo_latex_safe}

\end{{document}}
"""
                        output_tex = f"{final_name}.tex"
                        with open(output_tex, "w", encoding="utf-8") as f:
                            f.write(latex_template)
                            
                        status.update(label="4/4. Compilazione del PDF tramite pdflatex...", state="running")
                        
                        completato = subprocess.run(["pdflatex", "-interaction=nonstopmode", output_tex], capture_output=True, text=True)
                        
                        if completato.returncode != 0:
                            st.warning("Ci sono stati avvertimenti o errori lievi durante la compilazione LaTeX, non bloccanti.")
                            with st.expander("Vedi Log pdflatex"):
                                st.text(completato.stdout)
                                
                        pdf_filename = f"{final_name}.pdf"
                        md_filename = f"{final_name}.md"
                        
                        if os.path.exists(pdf_filename):
                            status.update(label="Completato con successo!", state="complete")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                with open(pdf_filename, "rb") as f:
                                    pdf_bytes = f.read()
                                st.download_button("Scarica PDF Formattato", pdf_bytes, file_name=pdf_filename, mime="application/pdf")
                            with col2:
                                st.download_button("Scarica File .tex", latex_template, file_name=output_tex, mime="text/plain")
                            with col3:
                                st.download_button("Scarica Markdown", testo_migliorato, file_name=md_filename, mime="text/markdown")
                                
                            # Salvataggio automatico in Archivio
                            ARCHIVE_DIR = "archivio_lezioni"
                            safe_materia = materia_trascrizione.replace(' ', '_').replace('/', '_')
                            MATERIA_DIR = os.path.join(ARCHIVE_DIR, safe_materia)
                            os.makedirs(MATERIA_DIR, exist_ok=True)
                            base_filename = f"{final_name}_{int(time.time())}"
                            
                            # Salva anche il PDF nell'archivio
                            pdf_archive_path = os.path.join(MATERIA_DIR, f"{base_filename}.pdf")
                            shutil.copy2(pdf_filename, pdf_archive_path)
                            
                            archive_data = {
                                "titolo": titolo_doc,
                                "materia": materia_trascrizione,
                                "testo_grezzo": testo_grezzo,
                                "testo_migliorato": testo_migliorato,
                                "data": data_lezione.strftime("%Y-%m-%d")
                            }
                            
                            with open(os.path.join(MATERIA_DIR, f"{base_filename}.json"), "w", encoding="utf-8") as f:
                                json.dump(archive_data, f, ensure_ascii=False, indent=4)
                            
                            st.session_state["ultima_lezione"] = archive_data
                            
                            st.divider()
                            st.subheader("🎵 Riascolta l'audio")
                            st.audio(uploaded_audio)
                            
                        else:
                            raise FileNotFoundError("Il PDF non è stato generato. Errore pdflatex.")
                            
                        for ext in [".aux", ".log", ".out"]:
                            try: os.remove(f"{final_name}{ext}")
                            except OSError: pass
                        
                    except Exception as e:
                        status.update(label="Errore nell'elaborazione", state="error")
                        st.error(f"Si è verificato un errore: {e}")

with tab_studio:
    st.header("🎓 Strumenti di Studio: Quiz, Riassunti e Flashcards")
    
    ARCHIVE_DIR = "archivio_lezioni"
    files_studio = []
    if os.path.exists(ARCHIVE_DIR):
        for root, dirs, filenames in os.walk(ARCHIVE_DIR):
            for filename in filenames:
                if filename.endswith('.json'):
                    files_studio.append(os.path.join(root, filename))
                    
    if files_studio:
        nomi_belli_studio = {}
        for path_f in files_studio:
            materia_dir = os.path.basename(os.path.dirname(path_f))
            try:
                with open(path_f, "r", encoding="utf-8") as filejson:
                    d = json.load(filejson)
                    title_full = d.get("titolo", "Senza nome")
                    date_val = d.get("data", "")
                    nomi_belli_studio[path_f] = f"[{materia_dir.replace('_', ' ')}] {date_val} — {title_full}"
            except:
                nomi_belli_studio[path_f] = os.path.basename(path_f)
                
        files_studio.sort(reverse=True, key=lambda x: os.path.getmtime(x) if os.path.exists(x) else x)
        
        opzioni = ["--- Seleziona una lezione dall'archivio ---"] + files_studio
        selezione = st.selectbox("Carica lezione rapida:", opzioni, format_func=lambda x: nomi_belli_studio.get(x, x) if x != opzioni[0] else x)
        
        if selezione != opzioni[0]:
            try:
                with open(selezione, "r", encoding="utf-8") as filejson:
                    st.session_state["ultima_lezione"] = json.load(filejson)
            except Exception:
                pass

    if "ultima_lezione" not in st.session_state:
        st.info("💡 Suggerimento: Trascrivi prima una lezione nella tab 'Trascrizione', oppure seleziona una lezione dal menu a tendina qui sopra per generare materiali di studio creativi.")
    else:
        lezione = st.session_state["ultima_lezione"]
        st.success(f"**Lezione Attiva:** {lezione.get('titolo', 'Senza Titolo')}")
        
        col_s1, col_s2 = st.columns([1, 2])
        with col_s1:
            st.subheader("Cosa vuoi generare?")
            tipo_materiale = st.radio("Seleziona Strumento:", [
                "🎙️ Script Podcast a 2 Voci (Stile NotebookLM)",
                "🗂 Flashcards (Domanda & Risposta per Anki/Quizlet)", 
                "📝 Riassunto Breve (Bullet points)",
                "✅ Quiz a risposta multipla (con correttore)",
                "🗺 Mappa Concettuale (testuale e markdown)",
                "⏱ Punti Chiave (Da ripassare prima dell'esame)"
            ])
            
            generazione_btn = st.button("✨ Genera Materiale con AI", type="primary", use_container_width=True)
            
        with col_s2:
            if generazione_btn:
                if not os.getenv("GEMINI_API_KEY") and not api_key_input:
                    st.error("⚠️ Inserisci la Gemini API Key nella barra laterale o nel file .env per poter generare i materiali di studio.")
                else:
                    with st.spinner(f"Sto elaborando '{tipo_materiale}' basandomi sui tuoi appunti..."):
                        client = genai.Client(api_key=api_key_input or os.getenv("GEMINI_API_KEY"))
                        
                        is_interactive = "Flashcards" in tipo_materiale or "Quiz" in tipo_materiale
                        
                        if "Flashcards" in tipo_materiale:
                            prompt_studio = f"""Agisci come un'intelligenza artificiale di sintesi analitica ispirata a NotebookLM.
                            Estrai i concetti chiave dai seguenti appunti e genera un set di Flashcards.
                            Devi rispondere ESCLUSIVAMENTE con un array JSON puro nel seguente formato:
                            [
                                {{"domanda": "Qual è la definizione di...", "risposta": "La definizione è..."}}
                            ]
                            Assicurati che non ci sia formattazione markdown (niente ```json) prima o dopo l'array.
                            
                            --- APPUNTI ---
                            {lezione['testo_migliorato']}
                            """
                        elif "Quiz" in tipo_materiale:
                            prompt_studio = f"""Agisci come un'intelligenza artificiale di sintesi analitica ispirata a NotebookLM.
                            Genera un Quiz a risposta multipla basato ESCLUSIVAMENTE sui seguenti appunti.
                            Devi rispondere ESCLUSIVAMENTE con un array JSON puro nel seguente formato:
                            [
                                {{
                                    "domanda": "Quale delle seguenti...", 
                                    "opzioni": ["Opzione A", "Opzione B", "Opzione C", "Opzione D"], 
                                    "corretta": 1, 
                                    "spiegazione": "L'opzione B è corretta perché..."
                                }}
                            ]
                            L'indice 'corretta' deve essere un intero da 0 a 3, che indica l'opzione giusta. Niente markdown.
                            
                            --- APPUNTI ---
                            {lezione['testo_migliorato']}
                            """
                        elif "Podcast" in tipo_materiale:
                            prompt_studio = f"""Agisci come il motore "Audio Overview" di NotebookLM. Il tuo compito è creare un copione per un podcast a 2 voci (Leo e Sofia) basato ESCLUSIVAMENTE sui seguenti appunti della lezione.
                            
                            REGOLE FONDAMENTALI (STILE NotebookLM):
                            1. GROUNDING ASSOLUTO: Non inventare o aggiungere fatti esterni, limitati a discutere in profondità i concetti e le formule presenti negli appunti.
                            2. DINAMICA DEI CONDUTTORI: Leo è il presentatore curioso che fa domande acute da "studente modello", Sofia è l'esperta appassionata che usa analogie brillanti e spiega i concetti in modo super chiaro e discorsivo.
                            3. CONVERSAZIONE NATURALE: Usa un tono radiofonico, avvincente e informale (es. "Aspetta, mi stai dicendo che...", "Esattamente, Leo!"). Devono esserci stupore e connessioni logiche chiare.
                            4. FORMATO PULITO: Scrivi ESCLUSIVAMENTE il testo delle battute. Non usare annotazioni di regia (es. [ride], [pausa]) né grassetto, in modo che un sintetizzatore vocale le legga fluidamente come testo normale.
                            
                            --- APPUNTI DELLA LEZIONE ---
                            {lezione['testo_migliorato']}
                            """
                        else:
                            prompt_studio = f"""Agisci come un'intelligenza artificiale di sintesi analitica di altissimo livello, ispirata al motore di NotebookLM di Google. 
                            Il tuo obiettivo è trasformare gli appunti forniti in: {tipo_materiale}. 
                            
                            REGOLE FONDAMENTALI (STILE NotebookLM):
                            1. GROUNDING ASSOLUTO: Basati ESCLUSIVAMENTE sui concetti, fatti e dati presenti nel testo fornito. Non inventare, allucinare o aggiungere conoscenze esterne, anche se corrette. Se un'informazione non c'è, omettila.
                            2. SINTESI PROFONDA E INTUITIVA: Non fare un semplice copia-incolla riassuntivo. Estrapola i temi portanti, evidenzia i collegamenti logici tra i concetti e ristruttura le informazioni in modo che siano istantaneamente comprensibili e facili da studiare e memorizzare.
                            3. CHIAREZZA ESPOSITIVA: Usa un tono accademico ma estremamente accessibile e stimolante.
                            
                            REGOLE DI FORMATTAZIONE:
                            L'intero output DEVE ESSERE UN CODICE LaTeX VALIDO per il corpo di un documento (niente preambolo). 
                            Usa \\begin{{itemize}} e \\item per gli elenchi, non usare mai formattazioni Markdown.
                            Usa rigorosamente il formato LaTeX per le formule matematiche (usando $...$).
                            NON usare mai il grassetto (niente ** o \\textbf).
                            
                            --- APPUNTI DELLA LEZIONE DA STUDIARE ---
                            {lezione['testo_migliorato']}
                            """
                            
                        try:
                            if is_interactive:
                                from google.genai import types
                                import json
                                conf = types.GenerateContentConfig(response_mime_type="application/json")
                                risposta_studio = client.models.generate_content(
                                    model=gemini_model,
                                    contents=prompt_studio,
                                    config=conf
                                )
                                try:
                                    parsed_data = json.loads(risposta_studio.text)
                                    st.session_state["live_studio_data"] = parsed_data
                                    st.session_state["live_studio_type"] = "Flashcards" if "Flashcards" in tipo_materiale else "Quiz"
                                    st.session_state["flashcard_idx"] = 0
                                    st.session_state["flashcard_flipped"] = False
                                    st.session_state["quiz_submitted"] = False
                                    for key in list(st.session_state.keys()):
                                        if key.startswith("quiz_q_"):
                                            del st.session_state[key]
                                    st.session_state.pop("static_studio_html", None)
                                except Exception as json_e:
                                    st.error(f"Errore nel parsing JSON del modello: {json_e}")
                            else:
                                risposta_studio = client.models.generate_content(
                                    model=gemini_model,
                                    contents=prompt_studio
                                )
                                st.session_state["live_studio_data"] = None
                                st.session_state["static_studio_html"] = risposta_studio.text
                                st.markdown(risposta_studio.text)
                                
                                if "Podcast" in tipo_materiale:
                                    st.info("🎧 Generazione dell'audio in corso... (potrebbe richiedere un po' di tempo)")
                                    from gtts import gTTS
                                    l_audio = lingua_audio if lingua_audio in ["it", "en"] else "it"
                                    tts = gTTS(text=risposta_studio.text, lang=l_audio, slow=False)
                                    audio_file = "Studio_Podcast.mp3"
                                    tts.save(audio_file)
                                    
                                    with open(audio_file, "rb") as f:
                                        st.download_button(
                                            "📥 Scarica Podcast (.mp3)", 
                                            f.read(), 
                                            file_name=f"Studio_Podcast_{lezione.get('materia', 'Generale')}.mp3", 
                                            mime="audio/mpeg"
                                        )
                                    try: os.remove(audio_file)
                                    except OSError: pass
                                else:
                                    titolo_s = f"{tipo_materiale.split()[1]} di {lezione.get('materia', 'Generale')}"
                                    latex_template = fr"""\documentclass{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage{{geometry}}
\usepackage{{hyperref}}
\geometry{{a4paper, margin=1in}}

\title{{{escape_latex(titolo_s)}}}
\author{{Generato da AI Pipeline}}
\date{{\today}}

\begin{{document}}
\maketitle

{risposta_studio.text}

\end{{document}}
"""
                                    tmp_tex = "studio_out.tex"
                                    with open(tmp_tex, "w", encoding="utf-8") as f:
                                        f.write(latex_template)
                                    subprocess.run(["pdflatex", "-interaction=nonstopmode", tmp_tex], capture_output=True, text=True)
                                    
                                    tmp_pdf = "studio_out.pdf"
                                    if os.path.exists(tmp_pdf):
                                        with open(tmp_pdf, "rb") as f:
                                            st.download_button(
                                                "📥 Scarica Documento (.pdf)", 
                                                f.read(), 
                                                file_name=f"Studio_{tipo_materiale.split()[1]}_{lezione.get('materia', 'Generale')}.pdf", 
                                                mime="application/pdf"
                                            )
                                    else:
                                        st.warning("Compilazione PDF fallita, fallback su Markdown.")
                                        st.download_button(
                                            "📥 Scarica Materiale Generato (.md)", 
                                            risposta_studio.text, 
                                            file_name=f"Studio_{tipo_materiale.split()[1]}_{lezione.get('materia', 'Generale')}.md", 
                                            mime="text/markdown"
                                        )
                                        
                                    for ext in [".aux", ".log", ".out", ".tex", ".pdf"]:
                                        try: os.remove(f"studio_out{ext}")
                                        except OSError: pass
                        except Exception as e:
                            st.error(f"Errore durante la generazione: {e}")

            # Rendering UI Interattiva (Fuori dal pulsante)
            if st.session_state.get("live_studio_data"):
                data = st.session_state["live_studio_data"]
                ltype = st.session_state["live_studio_type"]
                
                st.divider()
                st.subheader(f"⚡ {ltype} Live (NotebookLM Style)")
                
                if ltype == "Flashcards" and isinstance(data, list) and len(data) > 0:
                    idx = st.session_state.get("flashcard_idx", 0)
                    flipped = st.session_state.get("flashcard_flipped", False)
                    card = data[idx]
                    
                    st.markdown(f"### 🃏 Carta {idx+1} di {len(data)}")
                    st.info(f"**Domanda:**\n\n{card.get('domanda', '')}")
                    
                    if flipped:
                        st.success(f"**Risposta:**\n\n{card.get('risposta', '')}")
                    
                    bcol1, bcol2, bcol3 = st.columns([1, 2, 1])
                    with bcol1:
                        if st.button("⬅️ Precedente", disabled=(idx == 0), key="fc_prev"):
                            st.session_state["flashcard_idx"] = idx - 1
                            st.session_state["flashcard_flipped"] = False
                            st.rerun()
                    with bcol2:
                        if st.button("🔄 Gira Carta" if not flipped else "Nascondi Risposta", use_container_width=True, key="fc_flip"):
                            st.session_state["flashcard_flipped"] = not flipped
                            st.rerun()
                    with bcol3:
                        if st.button("Successiva ➡️", disabled=(idx == len(data) - 1), key="fc_next"):
                            st.session_state["flashcard_idx"] = idx + 1
                            st.session_state["flashcard_flipped"] = False
                            st.rerun()
                            
                elif ltype == "Quiz" and isinstance(data, list) and len(data) > 0:
                    submitted = st.session_state.get("quiz_submitted", False)
                    
                    for i, q in enumerate(data):
                        st.markdown(f"#### {i+1}. {q.get('domanda', '')}")
                        ops = q.get("opzioni", [])
                        
                        k = f"quiz_q_{i}"
                        scelta = st.radio("Seleziona:", ops, key=k, index=None, disabled=submitted, label_visibility="collapsed")
                        
                        if submitted:
                            corretta_idx = q.get("corretta", 0)
                            if corretta_idx < len(ops) and scelta == ops[corretta_idx]:
                                st.success(f"✅ Corretto! {q.get('spiegazione', '')}")
                            elif scelta is not None:
                                st.error(f"❌ Sbagliato! La risposta corretta era: **{ops[corretta_idx] if corretta_idx < len(ops) else 'N/A'}**")
                                st.info(f"💡 Spiegazione: {q.get('spiegazione', '')}")
                            else:
                                st.warning(f"⚠️ Non hai risposto. La risposta corretta era: **{ops[corretta_idx] if corretta_idx < len(ops) else 'N/A'}**")
                                st.info(f"💡 Spiegazione: {q.get('spiegazione', '')}")
                        st.markdown("---")
                        
                    if not submitted:
                        if st.button("✅ Verifica Tutte le Risposte", type="primary", use_container_width=True, key="quiz_verify"):
                            st.session_state["quiz_submitted"] = True
                            st.rerun()
                    else:
                        if st.button("🔄 Riprova Quiz", use_container_width=True, key="quiz_retry"):
                            st.session_state["quiz_submitted"] = False
                            for i in range(len(data)):
                                st.session_state[f"quiz_q_{i}"] = None
                            st.rerun()

with tab_chat:
    st.header("💬 Chat Globale (Stile NotebookLM)")
    
    ARCHIVE_DIR = "archivio_lezioni"
    files = []
    if os.path.exists(ARCHIVE_DIR):
        for root, dirs, filenames in os.walk(ARCHIVE_DIR):
            for filename in filenames:
                if filename.endswith('.json'):
                    files.append(os.path.join(root, filename))

    if not files and "ultima_lezione" not in st.session_state:
        st.info("💡 Suggerimento: Completa una trascrizione prima di usare la chat.")
    else:
        # Recupera nomi belli
        nomi_belli = {}
        for path_f in files:
            materia_dir = os.path.basename(os.path.dirname(path_f))
            try:
                with open(path_f, "r", encoding="utf-8") as filejson:
                    d = json.load(filejson)
                    title_full = d.get("titolo", "Senza nome")
                    date_val = d.get("data", "")
                    nomi_belli[path_f] = f"[{materia_dir.replace('_', ' ')}] {date_val} — {title_full}"
            except:
                nomi_belli[path_f] = os.path.basename(path_f)

        st.write("Questa chat agisce come una **Knowledge Base**. Seleziona una o più lezioni su cui l'AI deve basare le sue risposte:")
        
        default_files = []
        # Preseleziona l'ultima lezione se presente
        if "ultima_lezione" in st.session_state:
            # cerco il matching nell'archivio, ma se non è salvato è un mock. Per semplicità usiamo un fallback.
            st.success(f"📌 Lezione in memoria attiva: **{st.session_state['ultima_lezione'].get('titolo')}**")
            
        selected_files_for_chat = st.multiselect("Lezioni per il contesto:", files, format_func=lambda x: nomi_belli.get(x, x))
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt_chat := st.chat_input("Fai una domanda per farti spiegare meglio un concetto..."):
            st.session_state.messages.append({"role": "user", "content": prompt_chat})
            with st.chat_message("user"):
                st.markdown(prompt_chat)

            with st.chat_message("assistant"):
                if not os.getenv("GEMINI_API_KEY") and not api_key_input:
                    st.error("⚠️ Inserisci la Gemini API Key per usare la chat.")
                else:
                    # Raccogliamo il contesto
                    testo_contesto = ""
                    if "ultima_lezione" in st.session_state and not selected_files_for_chat:
                         testo_contesto += "\n\n--- Appunti in memoria ---\n" + st.session_state["ultima_lezione"].get("testo_migliorato", "")
                    
                    for pf in selected_files_for_chat:
                         try:
                             with open(pf, "r", encoding="utf-8") as filejson:
                                 d = json.load(filejson)
                                 testo_contesto += f"\n\n--- Lezione: {d.get('titolo')} ---\n{d.get('testo_migliorato', '')}"
                         except:
                             pass
                             
                    client = genai.Client(api_key=api_key_input or os.getenv("GEMINI_API_KEY"))
                    
                    system_prompt = f"Sei un assistente universitario esperto (Stile NotebookLM). Rispondi alle domande dello studente basandoti ESCLUSIVAMENTE sui seguenti appunti.\n\nCONTESTO:\n{testo_contesto}\n\nSe la risposta non è nel contesto o se non ci sono appunti scelti, dillo chiaramente spiegando cosa fare."
                    
                    history = []
                    for m in st.session_state.messages[:-1]:
                        role = "user" if m["role"] == "user" else "model"
                        history.append({"role": role, "parts": [{"text": m["content"]}]})
                    
                    chat_session = client.chats.create(model=gemini_model, history=history)
                    
                    risposta_ai = ""
                    placeholder = st.empty()
                    try:
                        resp = chat_session.send_message(f"Sistema: {system_prompt}\n\nDomanda Studente: {prompt_chat}")
                        risposta_ai = resp.text
                        placeholder.markdown(risposta_ai)
                    except Exception as e:
                        placeholder.error(f"Errore nella generazione della risposta: {e}")
                        
                    st.session_state.messages.append({"role": "assistant", "content": risposta_ai})
            
        if st.button("Pulisci Cronologia Chat testuale", key="clear_chat"):
            st.session_state.messages = []
            st.rerun()

with tab_archivio:
    st.header("🗂 Archivio Lezioni Locali")
    st.write("Tutti gli appunti generati vengono salvati in automatico per potervi accedere in seguito e ricaricarli negli strumenti di studio.")
    
    ARCHIVE_DIR = "archivio_lezioni"
    
    files = []
    if os.path.exists(ARCHIVE_DIR):
        for root, dirs, filenames in os.walk(ARCHIVE_DIR):
            for filename in filenames:
                if filename.endswith('.json'):
                    files.append(os.path.join(root, filename))

    if not files:
        st.info("L'archivio è vuoto. Le tue lezioni appariranno qui.")
    else:
        st.subheader("🔎 Ricerca Globale")
        search_query = st.text_input("Cerca parola chiave o concetto in tutte le lezioni...", placeholder="Es: termodinamica, neuroni...")
        if search_query:
            st.write(f"Risultati per: **{search_query}**")
            found_any = False
            for path_f in files:
                try:
                    with open(path_f, "r", encoding="utf-8") as filejson:
                        d = json.load(filejson)
                        if search_query.lower() in d.get("testo_migliorato", "").lower() or search_query.lower() in d.get("titolo", "").lower():
                            st.markdown(f"- **{d.get('titolo', 'Senza nome')}** ({d.get('materia', '')}): Trovato nel testo.")
                            found_any = True
                except:
                    pass
            if not found_any:
                st.warning("Nessun risultato trovato nell'archivio.")
        
        st.divider()

        files.sort(reverse=True, key=lambda x: os.path.getmtime(x) if os.path.exists(x) else x) # I più recenti prima
        
        # Dizionario per far apparire nomi carini nel Selezionatore
        nomi_belli = {}
        for path_f in files:
            materia_dir = os.path.basename(os.path.dirname(path_f))
            try:
                with open(path_f, "r", encoding="utf-8") as filejson:
                    d = json.load(filejson)
                    title_full = d.get("titolo", "Senza nome")
                    date_val = d.get("data", "")
                    nomi_belli[path_f] = f"[{materia_dir.replace('_', ' ')}] {date_val} — {title_full}"
            except:
                nomi_belli[path_f] = os.path.basename(path_f)
                
        selected_file = st.selectbox("Seleziona una lezione salvata", files, format_func=lambda x: nomi_belli.get(x, x))
        
        if selected_file:
            path_f = selected_file
            try:
                with open(path_f, "r", encoding="utf-8") as filejson:
                    dati_archivio = json.load(filejson)
                
                st.subheader(dati_archivio.get("titolo", "Senza Titolo"))
                st.caption(f"🗓 Data: **{dati_archivio.get('data', 'N/A')}** | 📚 Materia: **{dati_archivio.get('materia', 'N/A')}**")
                
                pdf_path = path_f.replace('.json', '.pdf')
                if not os.path.exists(pdf_path) and "testo_migliorato" in dati_archivio:
                    # Genera PDF mancante al volo
                    testo_latex_safe = escape_latex(dati_archivio["testo_migliorato"])
                    titolo_doc = dati_archivio.get("titolo", "Appunti della Lezione")
                    latex_template = fr"""\documentclass{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage{{geometry}}
\usepackage{{hyperref}}
\geometry{{a4paper, margin=1in}}

\title{{{escape_latex(titolo_doc)}}}
\author{{Generato da AI Pipeline}}
\date{{\today}}

\begin{{document}}
\maketitle

{testo_latex_safe}

\end{{document}}
"""
                    tmp_tex = "appunti_onfly.tex"
                    with open(tmp_tex, "w", encoding="utf-8") as f:
                        f.write(latex_template)
                    subprocess.run(["pdflatex", "-interaction=nonstopmode", tmp_tex], capture_output=True, text=True)
                    if os.path.exists("appunti_onfly.pdf"):
                        import shutil
                        shutil.copy2("appunti_onfly.pdf", pdf_path)
                        os.remove("appunti_onfly.pdf")
                    for ext in [".aux", ".log", ".out", ".tex"]:
                        try: os.remove(f"appunti_onfly{ext}")
                        except OSError: pass

                col_a1, col_a2, col_a3 = st.columns(3)
                with col_a1:
                    if st.button("🧠 Carica in Strumenti di Studio", use_container_width=True, type="primary"):
                        st.session_state["ultima_lezione"] = dati_archivio
                        st.success("Lezione ricaricata in memoria! Usa la tab in alto 'Strumenti di Studio'.")
                with col_a2:
                    if os.path.exists(pdf_path):
                        with open(pdf_path, "rb") as f:
                            st.download_button("📥 Scarica di nuovo gli appunti", f.read(), file_name=os.path.basename(pdf_path), use_container_width=True, mime="application/pdf")
                    else:
                        st.download_button("📥 Scarica di nuovo gli appunti", dati_archivio.get("testo_migliorato", ""), file_name=f"{selected_file.replace('.json', '')}.md", use_container_width=True, mime="text/markdown")
                with col_a3:
                    if st.button("🔄 Ritenta Miglioramento Gemini", use_container_width=True):
                        if not api_key_input and not os.getenv("GEMINI_API_KEY"):
                            st.error("Inserisci la Gemini API Key!")
                        else:
                            with st.spinner("Miglioramento in corso..."):
                                try:
                                    client = genai.Client(api_key=api_key_input or os.getenv("GEMINI_API_KEY"))
                                    materia_arch = dati_archivio.get('materia', 'Generale')
                                    inc_glossary_arch = materie_db.get(materia_arch, "") if materia_arch != "Nessuna" else ""
                                    base_prompt = f"""Agisci come un revisore accademico esperto e un trascrittore professionale.
Usa il seguente testo grezzo, trascritto automaticamente da una lezione universitaria (materia: {materia_arch}), e miglioralo applicando le seguenti regole rigorose:
1. Correggi TUTTI gli errori di ortografia, punteggiatura, sintassi.
2. Elimina false partenze, balbettii e riempitivi tipici del parlato ("eh", "cioè").
3. Deduci e correggi i termini tecnici o accademici che il sistema speech-to-text potrebbe aver frainteso.
{"!! IMPORTANTE: Usa OBBLIGATORIAMENTE questo GLOSSARIO DELLA MATERIA per risolvere ambiguita' fonetiche: " + inc_glossary_arch if inc_glossary_arch else ""}
4. Dividi il testo in paragrafi logici ben strutturati per migliorarne la leggibilità.
5. Inserisci punti elenco se il professore elenca degli elementi.
6. L'intero output deve essere un CODICE LaTeX VALIDO per il corpo del documento. Usa \begin{itemize} e \item per gli elenchi, NON usare i trattini del Markdown.
7. Le formule matematiche devono essere scritte RIGOROSAMENTE in sintassi LaTeX (usando $...$ per inline o \[...\] per blocchi). Assicurati di usare comandi validi.
8. ASSOLUTAMENTE CRITICO: Le parole del professore NON devono essere assolutamente riassunte o tagliate. Devono solamente essere corrette (sintassi, punteggiatura) e, in caso, implementate e integrate da formule in LaTeX, diagrammi o concetti estratti direttamente dai PDF/immagini allegati alla lezione.
9. ASSOLUTAMENTE CRITICO: NON tradurre il testo per nessun motivo. Mantieni la LINGUA ORIGINALE in cui è stata trascritta la lezione.
10. Restituisci ESCLUSIVAMENTE il corpo testuale in LaTeX pronto per essere compilato (niente preamble, niente formattazione markdown).
"""
                                    testo_grezzo = dati_archivio.get("testo_grezzo", "")
                                    prompt = f"{base_prompt}\nTesto originale:\n{testo_grezzo}\n"
                                    response = client.models.generate_content(
                                        model=gemini_model,
                                        contents=prompt
                                    )
                                    nuovo_testo = response.text
                                    
                                    dati_archivio["testo_migliorato"] = nuovo_testo
                                    
                                    # Generazione LaTeX e PDF
                                    testo_latex_safe = escape_latex(nuovo_testo)
                                    titolo_doc = dati_archivio.get("titolo", "Appunti della Lezione")
                                    
                                    latex_template = fr"""\documentclass{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage{{geometry}}
\usepackage{{hyperref}}
\geometry{{a4paper, margin=1in}}

\title{{{escape_latex(titolo_doc)}}}
\author{{Generato da AI Pipeline}}
\date{{\today}}

\begin{{document}}
\maketitle

{testo_latex_safe}

\end{{document}}
"""
                                    tmp_tex = "appunti_ritenta.tex"
                                    with open(tmp_tex, "w", encoding="utf-8") as f:
                                        f.write(latex_template)
                                        
                                    subprocess.run(["pdflatex", "-interaction=nonstopmode", tmp_tex], capture_output=True, text=True)
                                    
                                    tmp_pdf = "appunti_ritenta.pdf"
                                    if os.path.exists(tmp_pdf):
                                        pdf_path_arch = path_f.replace('.json', '.pdf')
                                        import shutil
                                        shutil.copy2(tmp_pdf, pdf_path_arch)
                                        os.remove(tmp_pdf)
                                        
                                    for ext in [".aux", ".log", ".out", ".tex"]:
                                        try: os.remove(f"appunti_ritenta{ext}")
                                        except OSError: pass

                                    with open(path_f, "w", encoding="utf-8") as f:
                                        json.dump(dati_archivio, f, ensure_ascii=False, indent=4)
                                    st.success("Testo migliorato e PDF rigenerato con successo!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Errore: {e}")
                
                st.divider()
                st.markdown(dati_archivio.get("testo_migliorato", ""))
                
                with st.expander("Mostra Testo Grezzo Whisper"):
                    st.text(dati_archivio.get("testo_grezzo", ""))
            except Exception as e:
                st.error(f"Errore nella lettura del file: {e}")

st.divider()
st.info("💡 **Nota Ambiente**: Assicurati di avere `ffmpeg` e una distribuzione LaTeX (`pdflatex`) attivi sul tuo sistema.")