from . import aigvdbench, genbuster, genvideo, genvidbench

DATASETS = {m.NAME: m for m in (genvidbench, genbuster, aigvdbench, genvideo)}
