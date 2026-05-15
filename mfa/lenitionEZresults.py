import json
import nltk
from g2p_en import G2p
from lingpy.align.pairwise import nw_align
from rich.console import Console
from rich.table import Table

# Ensure NLTK data is present
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('punkt_tab', quiet=True)

g2p = G2p()
console = Console()

def get_canonical(text):
    """Converts Whisper text to a list of canonical IPA phonemes."""
    # Mapping Arpabet (G2P output) to clean IPA
    arpabet_to_ipa = {
        'AO': 'ɔ', 'AA': 'ɑ', 'IY': 'i', 'UW': 'u', 'EH': 'ɛ', 'IH': 'ɪ', 
        'AS': 'æ', 'AH': 'ʌ', 'ER': 'ɚ', 'AX': 'ə', 'EY': 'eɪ', 'AY': 'aɪ', 
        'OY': 'ɔɪ', 'AW': 'aʊ', 'OW': 'oʊ', 'P': 'p', 'B': 'b', 'T': 't', 
        'D': 'd', 'K': 'k', 'G': 'ɡ', 'CH': 'tʃ', 'JH': 'dʒ', 'F': 'f', 
        'V': 'v', 'TH': 'θ', 'DH': 'ð', 'S': 's', 'Z': 'z', 'SH': 'ʃ', 
        'ZH': 'ʒ', 'HH': 'h', 'M': 'm', 'N': 'n', 'NG': 'ŋ', 'L': 'l', 
        'R': 'ɹ', 'W': 'w', 'Y': 'j'
    }
    out = []
    # g2p handles punctuation and case internally
    for phoneme in g2p(text):
        base = ''.join([i for i in phoneme if not i.isdigit()]).strip()
        if base in arpabet_to_ipa:
            out.append(arpabet_to_ipa[base])
        elif base and base not in '.,!? ':
            out.append(base.lower())
    return out

# 1. Load your JSON
with open('analysis.json', 'r') as f:
    data = json.load(f)

# 2. Setup the Visualizer
table = Table(title="Whisper Canonical vs. Realized Phonetic Alignment", show_lines=True)
table.add_column("Time", style="magenta", justify="center")
table.add_column("Whisper Text", style="white", width=30)
table.add_column("Phonetic Alignment (Green: Canonical | Cyan: Realized)", ratio=1)

for entry in data:
    text_in = entry.get('text', '')
    ipa_in = entry.get('ipa', '')
    
    # Generate canonical from Whisper text
    canonical = get_canonical(text_in)
    # Tokenize realized IPA from your transcription
    realized = [p for p in ipa_in.strip().split() if p]
    
    if not canonical or not realized:
        continue

    try:
        # The Magic: Needleman-Wunsch Alignment
        alA, alB, score = nw_align(canonical, realized)
        
        # Colorize differences for "Lenition Spotting"
        top_str = ""
        bot_str = ""
        for c, r in zip(alA, alB):
            if c == r:
                top_str += f"[green]{c}[/green] "
                bot_str += f"[cyan]{r}[/cyan] "
            else:
                # Highlight lenition/changes in yellow/red
                top_str += f"[yellow]{c}[/yellow] "
                bot_str += f"[bold red]{r}[/bold red] "

        table.add_row(
            f"{entry.get('time'):.2f}",
            text_in,
            f"{top_str}\n{bot_str}"
        )
    except Exception as e:
        console.print(f"[red]Error on entry {entry.get('time')}: {e}[/red]")

console.print(table)
