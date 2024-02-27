% This is  testcbst.bst  created by cbst program
%           2001/06/06
% BibTeX `base' is derived from 'plain.bst'
% version 0.99b for BibTeX versions 0.99a or later, LaTeX version 2.09.
% Copyright (C) 1985, all rights reserved.
% Copying of this file is authorized only if either
% (1) you make absolutely no changes to your copy, including name, or
% (2) if you do make changes, you name it something other than
% btxbst.doc, plain.bst, unsrt.bst, alpha.bst, and abbrv.bst.
% This restriction helps ensure that all standard styles are identical.
% The file btxbst.doc has the documentation for this style.
%
% This file is a part of cbst and is modified from plain.bst 
%   by Level3 
%     hagihira@anes.med.osaka-u.ac.jp / pgb00174@nifty.ne.jp
%   Department of Anesthesiology, 
%   Osaka Prefectural Habikino Hospital,
%   3-7-1, Habikino, Habikino, Osaka 583-8588, Japan.
%
% for customizing bst files
%    1997.11.24  Ver 0.81
%    1997.11.26  Ver 0.82  added bold
%    1998.04.10  Ver 0.83  added output2
%    1998.04.15  Ver 0.84  added output.pre, output.post, output.pre.post
%    1998.04.21  Ver 0.85  output2 adapted for mktype Ver0.99
%    1999.02.02  Ver 0.86  added pinit, remove output2

% $Id: syori.bst,v 1.1.1.1 2001/07/09 06:05:31 ftani Exp $

% Fixing history of syori.bst/prml.bst
% 2007.05.23 (Hanada) added "inproceedings" field
% 2008.04.17 (Hanada) renamed to "prml.bst"
% 2013.10.17 (Hanada) Changed volume/number format (from "VOL, NUM" to "VOL-NUM"), added "conference" field (alias for "inproceedings")

ENTRY
  { address
    author
    booktitle
    chapter
    edition
    editor
    howpublished
    institution
    journal
    key
    month
    note
    number
    organization
    pages
    publisher
    school
    series
    title
    type
    volume
    year
  }
  { }
  { label }

INTEGERS { output.state before.all mid.sentence after.sentence after.block }

FUNCTION {init.state.consts}
{ #0 'before.all :=
  #1 'mid.sentence :=
  #2 'after.sentence :=
  #3 'after.block :=
}

STRINGS { s t }

% added by Level3
% for working
STRINGS { separator pre pre2 post post2 str }

FUNCTION {pinit}
{
  ", " 'separator :=
  "" 'pre :=
  "" 'post :=
  "" 'pre2 :=
  "" 'post2 :=
  "" 'str :=
}

% This needs "separator"
FUNCTION {output.nonnull}
{ 's :=
  output.state mid.sentence =
    { separator * write$ }
    { output.state after.block =
	{ add.period$ write$
	  newline$
	  "\newblock " write$
	}
	{ output.state before.all =
	    'write$
	    { add.period$ " " * write$ }
	  if$
	}
      if$
      mid.sentence 'output.state :=
    }
  if$
  pinit
  s
}

% This needs 'separator' if nessesary
FUNCTION {output}
{ duplicate$ empty$
    'pop$
    'output.nonnull
  if$
  pinit
}

% This needs 'separator' if nessesary
FUNCTION {output.pre}
{ duplicate$ empty$
    { pop$ pop$ }
    { * output.nonnull }
  if$
  pinit
}

% This needs 'separator' if nessesary
FUNCTION {output.post}
{ duplicate$ empty$
    { pop$ pop$ }
    { swap$ * output.nonnull }
  if$
  pinit
}

% This needs 'separator' if nessesary
FUNCTION {output.pre.post}
{ duplicate$ empty$
    { pop$ pop$ pop$ }
    { swap$ * * output.nonnull }
  if$
  pinit
}

% This needs 'separator' if nessesary
FUNCTION {output.check}
{ 't :=
  duplicate$ empty$
    { pop$ "empty " t * " in " * cite$ * warning$ }
    'output.nonnull
  if$
  pinit
}

FUNCTION {output.bibitem}
{ newline$
  "\bibitem{" write$
  cite$ write$
  "}" write$
  newline$
  ""
  before.all 'output.state :=
}

FUNCTION {fin.entry}
{ add.period$
  write$
  newline$
}

FUNCTION {new.block}
{ output.state before.all =
    'skip$
    { after.block 'output.state := }
  if$
}

FUNCTION {new.sentence}
{ output.state after.block =
    'skip$
    { output.state before.all =
	'skip$
	{ after.sentence 'output.state := }
      if$
    }
  if$
}

FUNCTION {not}
{   { #0 }
    { #1 }
  if$
}

FUNCTION {and}
{   'skip$
    { pop$ #0 }
  if$
}

FUNCTION {or}
{   { pop$ #1 }
    'skip$
  if$
}


FUNCTION {new.block.checka}
{ empty$
    'skip$
    'new.block
  if$
}

FUNCTION {new.block.checkb}
{ empty$
  swap$ empty$
  and
    'skip$
    'new.block
  if$
}

FUNCTION {new.sentence.checka}
{ empty$
    'skip$
    'new.sentence
  if$
}

FUNCTION {new.sentence.checkb}
{ empty$
  swap$ empty$
  and
    'skip$
    'new.sentence
  if$
}

FUNCTION {field.or.null}
{ duplicate$ empty$
    { pop$ "" }
    'skip$
  if$
}

FUNCTION {emphasize}
{ duplicate$ empty$
    { pop$ "" }
    { "{\em " swap$ * "}" * }
  if$
}

FUNCTION {bold}
{ duplicate$ empty$
    { pop$ "" }
    { "{\bf " swap$ * "}" * }
  if$
}

INTEGERS { nameptr namesleft numnames }

% This needs 'str'
%  in plain.bst
%  str := " and "
% 2021/11/17 kume "when the number of authers >= 4, add 'et al' after 1st author."
FUNCTION {format.names}
{ 's :=
  #1 'nameptr :=
  s num.names$ 'numnames :=
  numnames 'namesleft :=
  numnames #3 >
  {
  #2 'namesleft :=
  { namesleft #0 > }
  { 
	% format name
	s nameptr "{ff}" format.name$ is.kanji.str$
		{ % syori format
		  s nameptr "{ff}" format.name$ text.length$ #2 >
			{
				s nameptr "{ll}" format.name$ text.length$ #2 >
					{ s nameptr "{ff}{ll}" format.name$ 't := }
					{ s nameptr "{ff} {ll}" format.name$ 't :=}
				if$
			}
			{ s nameptr "{ff} {ll}" format.name$ 't := }
		  if$
		}
		{ s nameptr "{f.~}{v~}{ll}{, jj}" format.name$ 't := }
	if$
	nameptr #1 >
          { namesleft #1 >
            { ", " * t * }
            { " {\it et~al}." * }
          if$
        }
        't
       if$
       nameptr #1 + 'nameptr :=
       namesleft #1 - 'namesleft :=
       }
    while$
    }
    {
        { namesleft #0 > }
        { 
	    % format name
	    s nameptr "{ff}" format.name$ is.kanji.str$
		    { % syori format
		      s nameptr "{ff}" format.name$ text.length$ #2 >
			    {
				    s nameptr "{ll}" format.name$ text.length$ #2 >
					    { s nameptr "{ff}{ll}" format.name$ 't := }
					    { s nameptr "{ff} {ll}" format.name$ 't :=}
				    if$
			    }
			    { s nameptr "{ff} {ll}" format.name$ 't := }
		      if$
		    }
		    { s nameptr "{f.~}{v~}{ll}{, jj}" format.name$ 't := }
	    if$

	    % between author's name
	    nameptr #1 >
		    { namesleft #1 >
			    { s is.kanji.str$
				    { "\UTF{00A1}\UTF{0160}" * t * }
				    {", " * t * }
			      if$
			    }
			{ numnames #2 >
				{ "" * }
				'skip$
			  if$
			  t "others" = 
		            { " et~al" * }
			     { s is.kanji.str$
			         { "\UTF{00A1}\UTF{0160}" * t * }
				  { " and " * t * }
				  if$
			     }
			    if$
			  }
	               if$
			}
		't
	    if$
          nameptr #1 + 'nameptr :=
          namesleft #1 - 'namesleft :=
        }
      while$
     }
  if$
}

FUNCTION {format.authors}
{
  author empty$
    { "" }
    { author format.names }
  if$
}

% This needs 'pre', 'pre2', 'post', 'post2' and 'str'
%  In plain.bst 
%   pre  := "" post  := ", editor"
%   pre2 := "" post2 := ", editors"
%   str := " and "
FUNCTION {format.editors}
{ editor empty$
    { "" }
    { editor format.names
      editor num.names$ #1 >
	{ pre2 swap$ * post2 * }
	{ pre swap$ * post * }
      if$
    }
  if$
}

FUNCTION {format.title}
{ title
}

FUNCTION {n.dashify}
{ 't :=
  ""
    { t empty$ not }
    { t #1 #1 substring$ "-" =
	{ t #1 #2 substring$ "--" = not
	    { "--" *
	      t #2 global.max$ substring$ 't :=
	    }
	    {   { t #1 #1 substring$ "-" = }
		{ "-" *
		  t #2 global.max$ substring$ 't :=
		}
	      while$
	    }
	  if$
	}
	{ t #1 #1 substring$ *
	  t #2 global.max$ substring$ 't :=
	}
      if$
    }
  while$
}

FUNCTION {format.date}
{ year empty$
    { month empty$
	{ "" }
	{ "there's a month but no year in " cite$ * warning$
	  month
	}
      if$
    }
    { month empty$
	'year
	{ month " " * year * }
      if$
    }
  if$
}

% added by Level3
FUNCTION {format.str2}
{ 'str :=
  str empty$
  { "no" 'str * "item in " * cite$ * warning$
    ""
  }
  { str }
  if$
}

FUNCTION {tie.or.space.connect}
{ duplicate$ text.length$ #3 <
    { "~" }
    { " " }
  if$
  swap$ * *
}

FUNCTION {either.or.check}
{ empty$
    'pop$
    { "can't use both " swap$ * " fields in " * cite$ * warning$ }
  if$
}

%  This needs pre
%  in plain.bst
%  pre := "volume "
FUNCTION {format.bvolume}
{ volume empty$
    { "" }
    { pre volume tie.or.space.connect
      series empty$
	'skip$
	{ " of " * series emphasize * }
      if$
      "volume and number" number either.or.check
    }
  if$
}

%  in plain.bst
%  pre  := "number "
%  pre2 := "Number "
FUNCTION {format.number.series}
{ volume empty$
    { number empty$
	{ series field.or.null }
	{ output.state mid.sentence =
	    { pre }
	    { pre2 }
	  if$
	  number tie.or.space.connect
	  series empty$
	    { "there's a number but no series in " cite$ * warning$ }
	    { " in " * series * }
	  if$
	}
      if$
    }
    { "" }
  if$
}

% This needs 'pre' and 'post'
%  in plain.bst
%   pre := ""
%   post := " edition"
FUNCTION {format.edition2}
{ 'post :=
  edition empty$
    { "" }
    { output.state mid.sentence =
	{ edition "l" change.case$ pre swap$ * post * }
	{ edition "t" change.case$ pre swap$ * post * }
      if$
    }
  if$
}

INTEGERS { multiresult }

FUNCTION {multi.page.check}
{ 't :=
  #0 'multiresult :=
    { multiresult not
      t empty$ not
      and
    }
    { t #1 #1 substring$
      duplicate$ "-" =
      swap$ duplicate$ "," =
      swap$ "+" =
      or or
	{ #1 'multiresult := }
	{ t #2 global.max$ substring$ 't := }
      if$
    }
  while$
  multiresult
}

% This needs 'pre', 'pre2', 'post' and 'post2'
% in plain.bst
%  pre  := "page "  post  := ""
%  pre2 := "pages " post2 := ""
FUNCTION {format.pages}
{ pages empty$
    { "" }
    { pages multi.page.check
        { pre2 pages n.dashify tie.or.space.connect post2 *}
        { pre pages tie.or.space.connect post * }
      if$
    }
  if$
}

FUNCTION {format.pages.pre.pre2}
{ 'pre2 :=
  'pre :=
  pages empty$
    { "" }
    { pages multi.page.check
	{ pre2 pages n.dashify tie.or.space.connect post2 *}
	{ pre pages tie.or.space.connect post * }
      if$
    }
  if$
}

% This needs 'pre' and 'post'
% in plain.bst
%  pre  := "page "  post  := ""
FUNCTION {format.pages2}
{ pages empty$
    'skip$
    { duplicate$ empty$
	  { "" }
	  { pages n.dashify pre swap$ * post * }
      if$
    }
  if$
}

% This needs 'str'
% in plain.bst
%  str := ":"
%  pre := "("
%  post := ")"
FUNCTION {format.vol.num.pages2}
{ 'str :=
  volume field.or.null
  number empty$
    'skip$
    { pre number * post * *
      volume empty$
	{ "there's a number but no volume in " cite$ * warning$ }
	'skip$
      if$
    }
  if$
  pages empty$
    'skip$
    { duplicate$ empty$
	{ pop$ format.pages }
	{ str * pages n.dashify * }
      if$
    }
  if$
}

% This needs 'pre', 'pre2' in calling format.pages and 'str'
% in plain.bst 
%  str := "chapter "
FUNCTION {format.chapter.pages2}
{ 'str :=
  chapter empty$
    'format.pages
    { type empty$
	{ str }
	{ type "l" change.case$ }
      if$
      chapter tie.or.space.connect
      pages empty$
	'skip$
	{ ", " * format.pages * }
      if$
    }
  if$
}

% This needs 'pre'
% in plain.bst
%  separator := ", "
FUNCTION {format.in.ed.booktitle2}
{ 'str :=
  's :=
  booktitle empty$
    { title empty$
       { "" }
       { editor empty$
	  { s title emphasize * }
	  { s format.editors * separator * title emphasize * }
         if$
       }
      if$  
    }
    { editor empty$
	{ s booktitle emphasize * }
	{ s format.editors * separator * booktitle emphasize * }
      if$
    }
  if$
}

FUNCTION {empty.misc.check}
{ author empty$ title empty$ howpublished empty$
  month empty$ year empty$ note empty$
  and and and and and
  key empty$ not and
    { "all relevant fields are empty in " cite$ * warning$ }
    'skip$
  if$
}

FUNCTION {format.thesis.type}
{ type empty$
    'skip$
    { pop$
      type "t" change.case$
    }
  if$
}

% This needs 'str'
%  in plain.bst
%  str := "Technical Report"
FUNCTION {format.tr.number}
{ type empty$
    { "Technical Report" }
    'type
  if$
  number empty$
    { "t" change.case$ }
    { number tie.or.space.connect }
  if$
}

FUNCTION {format.article.crossref}
{ key empty$
    { journal empty$
	{ "need key or journal for " cite$ * " to crossref " * crossref *
	  warning$
	  ""
	}
	{ "In {\em " journal * "\/}" * }
      if$
    }
    { "In " key * }
  if$
  " \cite{" * crossref * "}" *
}

FUNCTION {format.crossref.editor}
{ editor #1 "{vv~}{ll}" format.name$
  editor num.names$ duplicate$
  #2 >
    { pop$ " et~al." * }
    { #2 <
	'skip$
	{ editor #2 "{ff }{vv }{ll}{ jj}" format.name$ "others" =
	    { " et~al." * }
	    { " and " * editor #2 "{vv~}{ll}" format.name$ * }
	  if$
	}
      if$
    }
  if$
}

FUNCTION {format.book.crossref}
{ volume empty$
    { "empty volume in " cite$ * "'s crossref of " * crossref * warning$
      "In "
    }
    { "Volume" volume tie.or.space.connect
      " of " *
    }
  if$
  editor empty$
  editor field.or.null author field.or.null =
  or
    { key empty$
	{ series empty$
	    { "need editor, key, or series for " cite$ * " to crossref " *
	      crossref * warning$
	      "" *
	    }
	    { "{\em " * series * "\/}" * }
	  if$
	}
	{ key * }
      if$
    }
    { format.crossref.editor * }
  if$
  " \cite{" * crossref * "}" *
}

FUNCTION {format.incoll.inproc.crossref}
{ editor empty$
  editor field.or.null author field.or.null =
  or
    { key empty$
	{ booktitle empty$
	    { "need editor, key, or booktitle for " cite$ * " to crossref " *
	      crossref * warning$
	      ""
	    }
	    { "In {\em " booktitle * "\/}" * }
	  if$
	}
	{ "In " key * }
      if$
    }
    { "In " format.crossref.editor * }
  if$
  " \cite{" * crossref * "}" *
}

%  write FUNCTIONs for each type such as article, book, inbook etc. below
%  here.
% -------------------------

FUNCTION {book}
{ output.bibitem
  %% FIXME
  author empty$
    'skip$
    { format.authors output } % output.check
  if$
  ", " 'separator :=
  title is.kanji.str$
    { title format.str2 "title" output.check }
    { title emphasize format.str2 "title" output.check }
  if$
  ". " 'separator :=
  publisher format.str2 "publisher" output.check
  ", " 'separator :=
  address  output
  ", " 'separator :=
  year format.str2 "year" output.check
  " " 'separator :=
  note output
  fin.entry
}

% 2009.10.25 'inbook' part added by H.Hanada
FUNCTION {inbook}
{ output.bibitem
  
  % show the authors
  format.authors "author" output.check
  ", " 'separator :=

  % show the editors if exist
  editor empty$
  {}{
    format.authors "editor" output.check
    "(ed.), " 'separator :=
  }
  if$
  
  % show the title
  format.title "title" output.check
  ". " 'separator :=
  
  % show information of the book
  "In chapter" output
  " " 'separator :=
  chapter format.str2 "chapter" output.check
  " " 'separator :=
  
  series missing$
  {}{
    "of" output
    " " 'separator :=
    series is.kanji.str$
      { series format.str2 "series" output.check }
      { series emphasize format.str2 "series" output.check }
    if$
    ", " 'separator :=
  }
  if$
  publisher format.str2 "publisher" output.check
  ", " 'separator :=
  address  output
  ", " 'separator :=
  year format.str2 "year" output.check
  
  pages n.dashify pre swap$ * post * output
  
  note missing$
  {}{ % if note field exists
    note output
  }
  if$
  
  fin.entry
}

% 2009.10.25 'mastersthesis' and 'phdthesis' part added by H.Hanada
FUNCTION {mastersthesis}
{
  output.bibitem

  % show the authors
  format.authors "author" output.check
  ", " 'separator :=
  
  % show the title
  "``" format.title * ".'' " * "title" output.check
  ", " 'separator :=
  
  % show the school
  school is.kanji.str$
  {
    % if the school name is Japanese
    school "\UTF{0153}\UTF{20AC}\UTF{00BB}\UTF{00CE}\UTF{00CF}\UTF{00C0}\UTF{00CA}\UTF{017E}" * format.str2 "school" output.check
    ", " 'separator :=
  }{
    % otherwise
    "Master's thesis of " school * format.str2 "school" output.check
    ", " 'separator :=
  }
  if$ 
  
  address output
  ", " 'separator :=
  
  % show the accepted date
  month missing$
  {
    % if the month is omitted
    year format.str2 "year" output.check
  }{
    % if the month is specified by number
    title is.kanji.str$
    {
      % if Japanese thesis
      year "\UTF{00C7}\UTF{00AF}" * output
      "" 'separator :=
      month "\UTF{00B7}\UTF{0081}E" * output
      "" 'separator :=
    }{
      % if the month is not specified by number
      month " " * year * output
    }
    if$
  }
  if$
  "." 'separator :=
  
  % show the note
  note missing$
  {}{ % if note field exists
    "(" note * ")" * output
  }
  if$
  
  fin.entry
}

FUNCTION {phdthesis}
{
  output.bibitem

  % show the authors
  format.authors "author" output.check
  ", " 'separator :=
  
  % show the title
  "``" format.title * ".''" * "title" output.check
  ", " 'separator :=
  
  % show the school
  type missing$
  {
    % if type is omitted
    school is.kanji.str$
    {
      % if the school name is Japanese
      school "\UTF{00C7}\UTF{00FB}×\UTF{00CE}\UTF{00CF}\UTF{00C0}\UTF{00CA}\UTF{017E}" * format.str2 "school" output.check
      ", " 'separator :=
    }{
      % otherwise
      "Ph. D. thesis of " school * format.str2 "school" output.check
      ", " 'separator :=
    }
    if$ 
  }{
    type " of " * school * format.str2 "school" output.check
    ", " 'separator :=
  }
  if$
  
  address output
  ", " 'separator :=
  
  % show the accepted date
  month missing$
  {
    % if the month is omitted
    year format.str2 "year" output.check
  }{
    % if the month is specified by number
    title is.kanji.str$
    {
      % if Japanese thesis
      year "\UTF{00C7}\UTF{00AF}" * output
      "" 'separator :=
      month "\UTF{00B7}\UTF{0081}E" * output
      "" 'separator :=
    }{
      % if the month is not specified by number
      month " " * year * output
    }
    if$
  }
  if$
  "." 'separator :=
  
  % show the note
  note missing$
  {}{ % if note field exists
    "(" note * ")" * output
  }
  if$
  
  fin.entry
}

% 2007.05.23 'inproceedings' part added by H.Hanada
% 2021.11.17 kume "add pre and post of title double quotation"
FUNCTION {conference}
{ output.bibitem
%  " and " format.authors2 "author" output.check
  %% FIXME
%  syorilang empty$
%      	{" and "   format.authors2 "author" output.check}
%	{"\UTF{008E}\UTF{00A1}\UTF{008E}\UTF{0160}"	format.authors2 "author" output.check}
%  if$
  format.authors "author" output.check
  ", " 'separator :=
  "``" format.title * ".''" * "title" output.check
  ", " 'separator :=
  % journal emphasize format.str2 "journal" output.check
  
  booktitle is.kanji.str$
	{ booktitle format.str2 "booktitle" output.check }
	{ booktitle emphasize format.str2 "booktitle" output.check }
  if$
  
  ", " 'separator :=
  volume missing$
  {
    year format.str2 "year" output.check
  }
  {
    number missing$
    {
      volume bold format.str2 "volume" output.check
      "" 'separator :=
      "(" year * ")" * format.str2 "year" output.check
    }
    {
      volume bold format.str2 "volume" output.check
      % Fixed by hanada on October 17, 2013
      %", " 'separator :=
      "-" 'separator :=
      
      number format.str2 "number" output.check
      "" 'separator :=
      "(" year * ")" * format.str2 "year" output.check
    }
    if$
  }
  if$
  
  pages empty$
    'skip$
    { pages n.dashify pre swap$ * post * output}
  if$
  
  % < added by Hanada, 2009.7.16
  note missing$
  {}{ % if note field exists
    "(" note * ")" * output
  }
  if$
  % >
  
  fin.entry
}

FUNCTION {inproceedings}
{ conference
}

% 2021.11.17 kume "add pre and post of title double quotation"
FUNCTION {article}
{ output.bibitem
%  " and " format.authors2 "author" output.check
  %% FIXME
%  syorilang empty$
%      	{" and "   format.authors2 "author" output.check}
%	{"\UTF{008E}\UTF{00A1}\UTF{008E}\UTF{0160}"	format.authors2 "author" output.check}
%  if$
  format.authors "author" output.check
  ", " 'separator :=
  "``" format.title * ".''" * "title" output.check
  ", " 'separator :=
  % journal emphasize format.str2 "journal" output.check
  journal is.kanji.str$
	{ journal format.str2 "journal" output.check }
	{ journal emphasize format.str2 "journal" output.check }
  if$
  ", " 'separator :=
  volume missing$
  {
    year format.str2 "year" output.check
  }
  {
    number missing$
    {
      volume bold format.str2 "volume" output.check
      "" 'separator :=
      "(" year * ")" * format.str2 "year" output.check
    }
    {
      volume bold format.str2 "volume" output.check
      % Fixed by hanada on October 17, 2013
      %", " 'separator :=
      "-" 'separator :=
      
      number format.str2 "number" output.check
      "" 'separator :=
      "(" year * ")" * format.str2 "year" output.check
    }
    if$
  }
  if$
%  volume bold format.str2 "volume" output.check
  "" 'separator :=
%  "(" year * ")" * format.str2 "year" output.check
  ", " 'separator :=

  format.pages2 output
  
  % < added by Hanada, 2009.5.15
  note missing$
  {}{ % if note field exists
    "(" note * ")" * output
  }
  if$
  % >
  
  fin.entry
}


FUNCTION {techreport}
{ output.bibitem
  %" and " format.authors2 "author" output.check
  format.authors "author" output.check	
  ", " 'separator :=
  format.title "title" output.check
  ",~" 'separator :=
  new.block
  format.tr.number emphasize output.nonnull
  institution "institution" output.check
  year format.str2 "year" output.check
  " " 'separator :=
  note output
  fin.entry
}

FUNCTION {misc}
{ output.bibitem
  %" and " format.authors2 output
  format.authors output
  title howpublished new.block.checkb
  format.title output
  howpublished new.block.checka
  howpublished output
  format.date output
  new.block
  note output
  fin.entry
  empty.misc.check
}

% -------------------------

FUNCTION {default.type} { misc }

MACRO {jan} {"January"}

MACRO {feb} {"February"}

MACRO {mar} {"March"}

MACRO {apr} {"April"}

MACRO {may} {"May"}

MACRO {jun} {"June"}

MACRO {jul} {"July"}

MACRO {aug} {"August"}

MACRO {sep} {"September"}

MACRO {oct} {"October"}

MACRO {nov} {"November"}

MACRO {dec} {"December"}

MACRO {acmcs} {"ACM Computing Surveys"}

MACRO {acta} {"Acta Informatica"}

MACRO {cacm} {"Communications of the ACM"}

MACRO {ibmjrd} {"IBM Journal of Research and Development"}

MACRO {ibmsj} {"IBM Systems Journal"}

MACRO {ieeese} {"IEEE Transactions on Software Engineering"}

MACRO {ieeetc} {"IEEE Transactions on Computers"}

MACRO {ieeetcad}
 {"IEEE Transactions on Computer-Aided Design of Integrated Circuits"}

MACRO {ipl} {"Information Processing Letters"}

MACRO {jacm} {"Journal of the ACM"}

MACRO {jcss} {"Journal of Computer and System Sciences"}

MACRO {scp} {"Science of Computer Programming"}

MACRO {sicomp} {"SIAM Journal on Computing"}

MACRO {tocs} {"ACM Transactions on Computer Systems"}

MACRO {tods} {"ACM Transactions on Database Systems"}

MACRO {tog} {"ACM Transactions on Graphics"}

MACRO {toms} {"ACM Transactions on Mathematical Software"}

MACRO {toois} {"ACM Transactions on Office Information Systems"}

MACRO {toplas} {"ACM Transactions on Programming Languages and Systems"}

MACRO {tcs} {"Theoretical Computer Science"}

MACRO {pr} {"Pattern Recognition"}

MACRO {prl} {"Pattern Recognition Letters"}

READ

STRINGS { longest.label }

INTEGERS { number.label longest.label.width }

FUNCTION {initialize.longest.label}
{ "" 'longest.label :=
  #1 'number.label :=
  #0 'longest.label.width :=
}

FUNCTION {longest.label.pass}
{ number.label int.to.str$ 'label :=
  number.label #1 + 'number.label :=
  label width$ longest.label.width >
    { label 'longest.label :=
      label width$ 'longest.label.width :=
    }
    'skip$
  if$
}

EXECUTE {initialize.longest.label}

ITERATE {longest.label.pass}

FUNCTION {begin.bib}
{ preamble$ empty$
    'skip$
    { preamble$ write$ newline$ }
  if$
  "\begin{thebibliography}{"  longest.label  * "}" * write$ newline$
}

EXECUTE {begin.bib}

EXECUTE {init.state.consts}

ITERATE {call.type$}

FUNCTION {end.bib}
{ newline$
  "\end{thebibliography}" write$ newline$
}

EXECUTE {end.bib}

% ------------ end of testcbst.bst --------------
