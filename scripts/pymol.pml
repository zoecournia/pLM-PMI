from pymol import cmd,stored

set depth_cue, 1
set fog_start, 0.4

set_color b_col, [36,36,85]
set_color t_col, [10,10,10]
set bg_rgb_bottom, b_col
set bg_rgb_top, t_col
set bg_gradient

set  spec_power  =  200
set  spec_refl   =  0

load "./Datasets/My_Dataset/Proteins_PDB/3see.pdb", protein
create ligands, protein and organic
select xlig, protein and organic
delete xlig

hide everything, all

color white, elem c
color bluewhite, protein
show_as cartoon, protein

show sticks, ligands
color magenta, ligands


select TP, chain A and resid 63 or chain A and resid 64 or chain A and resid 65 or chain A and resid 66 or chain A and resid 67 or chain A and resid 68 or chain A and resid 71 or chain A and resid 79 or chain A and resid 114 or chain A and resid 167 or chain A and resid 168
color green, TP
show licorice, TP

select FP, chain A and resid 55 or chain A and resid 59 or chain A and resid 61 or chain A and resid 69 or chain A and resid 72 or chain A and resid 77 or chain A and resid 82 or chain A and resid 83 or chain A and resid 84 or chain A and resid 115 or chain A and resid 169 or chain A and resid 219 or chain A and resid 222
show licorice, FP
color red, FP

select FN
show licorice, FN
color yellow, FN

deselect

orient
