@echo off
call C:\Users\aokuni\AppData\Local\miniforge3\Scripts\activate.bat fcad
python c:\github_repo\cad_automaton\src\proto2\poc.py > c:\github_repo\cad_automaton\output\poc_output.txt 2>&1
type c:\github_repo\cad_automaton\output\poc_output.txt
