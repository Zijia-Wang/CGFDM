clc
clear

parfile = 'params.json';
par = get_params(parfile);
outdir = par.OUT;

y = -39000
z = -100
var = 'Vs1'
[v, t] = get_fault_seismo(parfile,outdir,var,y,z);

figure
plot(t, v)
xlabel('T (sec')
