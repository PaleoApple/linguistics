import json
from g2p_en import G2p
from lingpy import Pairwise
from rich.console import Console
from rich.table import Table
from lingpy.align.pairwise import nw_align

# Initialize tools
g2p = G2p()
console = Console()

def get_canonical(text):
    # Converts "Wait a minute" -> ["W", "EY1", "T", ...] -> IPA
    # Simple mapping for common ARPABET to IPA
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
    for phoneme in g2p(text):
        base = ''.join([i for i in phoneme if not i.isdigit()]) # strip stress
        if base in arpabet_to_ipa:
            out.append(arpabet_to_ipa[base])
        elif base.isalpha():
            out.append(base.lower())
    return out

# Load your data
with open('analysis.json', 'r') as f:
    data = json.load(f)

# Create a Pretty Table
table = Table(title="Phonetic Realization Alignment")
table.add_column("Time", style="dim")
table.add_column("Alignment (Top: Canonical | Bottom: Realized)", style="bold")



for entry in data:
    # Safe get for text and ipa
    text_val = entry.get('text') or ""
    ipa_val = entry.get('ipa') or ""
    
    canonical = get_canonical(text_val)
    realized = [p for p in ipa_val.strip().split() if p]
    
    if not canonical or not realized:
        continue

    try:
        # nw_align returns (aligned_seqA, aligned_seqB, score)
        # It's robust and treats everything as strings
        alA, alB, score = nw_align(canonical, realized)
        
        top = "  ".join(alA)
        bottom = "  ".join(alB)
        
        table.add_row(
            str(entry['time']),
            f"[green]{top}[/green]\n[cyan]{bottom}[/cyan]"
        )
    except Exception as e:
        table.add_row(str(entry['time']), f"[red]Error: {str(e)}[/red]")
    
    table.add_section()

console.print(table)
