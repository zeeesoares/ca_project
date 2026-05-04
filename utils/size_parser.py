import re

def parse_size(size_str: str) -> int:
    """
    Transforma strings como '1.5 GB', '1024' ou '4MB' em inteiros (bytes).
    Aceita números puros (string ou int) e assume Bytes como padrão.
    """
    if isinstance(size_str, (int, float)):
        return int(size_str)

    s = str(size_str).strip().upper()
    
    match = re.match(r'^(\d+(?:\.\d+)?)\s*([A-Z]+)?$', s)
    
    if not match:
        raise ValueError(f"Formato de tamanho inválido: {size_str}")
        
    number_part, unit_part = match.groups()
    number = float(number_part)
    
    units = {
        'B': 1,
        'K': 1024, 'KB': 1024,
        'M': 1024**2, 'MB': 1024**2,
        'G': 1024**3, 'GB': 1024**3,
        'T': 1024**4, 'TB': 1024**4
    }
    
    multiplier = units.get(unit_part, 1) if unit_part else 1
    
    return int(number * multiplier)

def format_size(size_bytes: int) -> str:
    """
    Transform numbers (bytes) into readable strings like '1.00 GB'.
    """
    if size_bytes == 0:
        return "0B"
        
    units = ('B', 'KB', 'MB', 'GB', 'TB')
    i = 0
    while size_bytes >= 1024 and i < len(units) - 1:
        size_bytes /= 1024.0
        i += 1
        
    return f"{size_bytes:.2f}{units[i]}"