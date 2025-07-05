import re
from re import Pattern
from typing import Literal

patterns: dict[str, Pattern] = {}

stop_payment_pattern = r"(rtn\s+funds\s+held"
stop_payment_pattern += r"|Correction"
stop_payment_pattern += r"|User\s+Fees"
stop_payment_pattern += r"|Returned\s+Item\s+Stopped"
stop_payment_pattern += r"|RTN\s+NOT\s+AUTH"
stop_payment_pattern += r"|STOP\s+PAYMENT\s+FEE"
stop_payment_pattern += r"|RTN\s+STOP\s+PAYT)"
patterns['stop_payments'] = re.compile(stop_payment_pattern, flags=re.IGNORECASE)

nsf_pattern = r"(Cheque\s+returned\s+to\s+another\s+inst\s+NSF"
nsf_pattern += r"|\bNSF\b"
nsf_pattern += r"|Charge\s+returned\s+cheque"
nsf_pattern += r"|RTN\s+SPS"
nsf_pattern += r"|PAP\s+Return"
nsf_pattern += r"|RTN\s+FUNDS\s+HELD"
nsf_pattern += r"|Reversal"
nsf_pattern += r"|Effet\s+refus[eé]\s+sans\s+provision"
nsf_pattern += r"|NSF\s+return\s+Service\s+Charge"
nsf_pattern += r"|\s+Debit\s+Arrangement\s+AccountItem\s+returned\s+unpaid"
nsf_pattern += r"|CURRENT\s+CREDIT\s+ADJUSTMENT\s+EFT\s+REVERSAL"
nsf_pattern += r"|Internet\s+Banking\s+CORRECTION"
nsf_pattern += r"|returned\s+item\s+credit"
nsf_pattern += r"|Returned\s+(Item|cheque)[\s-]*NSF"
nsf_pattern += r"|AFT\s+Return\s+NSF"
nsf_pattern += r"|frais\s+opp\s+paiement"
nsf_pattern += r"|BR\.?0072)"
patterns['nsf'] = re.compile(nsf_pattern, flags=re.IGNORECASE)

lawyer_pattern  = r"(Boyle.Co"
lawyer_pattern += r"|Bingham\s+Law"
lawyer_pattern += r"|Duncan\s+Craig"
lawyer_pattern += r"|Mackenzie\s+Morgan\s+law"
lawyer_pattern += r"|Alberta\s+legal\s+Aid"
lawyer_pattern += r"|Duchin[\s,]+Bayda[\s&]+Kroczynski\s+Law\s+Firm"
lawyer_pattern += r"|Law\s+Society\s+of\s+Ontario"
lawyer_pattern += r"|Aird[\s&]+Berlis"
lawyer_pattern += r"|Fillmore\s+Riley"
lawyer_pattern += r"|Stein\s+Monast"
lawyer_pattern += r"|The\s+Advocates.Society)"
patterns['lawyer'] = re.compile(lawyer_pattern, flags=re.IGNORECASE)

def detect_pattern(pattern_key:Literal['stop_payments', 'nsf', 'lawyer'], trx_description:str, debit:float|None, credit:float|None) -> bool:
    return (patterns[pattern_key].search(trx_description) is not None)

