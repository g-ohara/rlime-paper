#!/usr/bin/env perl
$latex          = 'platex -synctex=1 -halt-on-error';
$latex_silent   = 'platex -synctex=1 -halt-on-error -interaction=batchmode';
$lualatex       = 'lualatex -shell-escape -synctex=1 -interaction=nonstopmode';
$bibtex         = 'pbibtex';
$dvipdf         = 'dvipdfmx %O -o %D %S';
$makeindex      = 'mendex %O -o %D %S';
$max_repeat     = 5;
$pdf_mode	    = 4; # generates pdf via dvipdfmx

# Prevent latexmk from removing PDF after typeset.
# This enables Skim to chase the update in PDF automatically.
$pvc_view_file_via_temporary = 0;

# Use Evince as a previewer
$pdf_previewer    = "evince";
