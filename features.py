import re
from urllib.parse import urlparse
import tldextract

def extract_url_features(url: str) -> dict:
    """Extracts unbiased lexical features matching the PhiUSIIL dataset."""
    clean_url = url.strip()
    parsed = urlparse(clean_url if "://" in clean_url else "http://" + clean_url)
    extracted = tldextract.extract(clean_url)
    
    domain = f"{extracted.domain}.{extracted.suffix}" if extracted.suffix else extracted.domain
    subdomain = extracted.subdomain

    # IP address detection in host
    host_only = parsed.netloc.split(":")[0]
    is_ip = 1 if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", host_only) else 0

    return {
        "DomainLength": len(domain),
        "IsDomainIP": is_ip,
        "NoOfSubDomain": len(subdomain.split(".")) if subdomain else 0,
        "IsHTTPS": 1 if parsed.scheme == "https" else 0
    }
