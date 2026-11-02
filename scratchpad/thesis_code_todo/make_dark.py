# Writes main_dark.tex (black page, white text) next to main.tex in the build directory. Not for submission.
import sys
d = sys.argv[1]
s = open(f'{d}/main.tex').read()
s = s.replace(r'\begin{document}', r'''\usepackage{xcolor}
\hypersetup{allcolors=cyan}
\lstset{rulecolor=\color{white}}
\begin{document}
\pagecolor{black}\color{white}
\makeatletter\let\default@color\current@color\makeatother''', 1).replace('{oxford_crest.jpeg}', '{oxford_crest_dark.jpeg}')
open(f'{d}/main_dark.tex', 'w').write(s)
