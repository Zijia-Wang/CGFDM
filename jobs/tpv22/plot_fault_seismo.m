clc
clear
addpath('../../matlab');
parfile = 'params.json';
par = get_params(parfile);
outdir = par.OUT;
nfault = 2;   % the station locate on which fault 

y = 10000
z = -10000
var = 'Vs1'
[v, t] = get_fault_seismo(parfile,outdir,var,y,z, nfault);

% EQDyna = load('EQdyna_10_10.txt');
% time = EQDyna(:,1);
% vs1 = EQDyna(:,3);
% 
% DayFD = load('DayFD_10_10.txt');
% time2 = DayFD(:,1);
% vs1_2 = DayFD(:,3);
% 
% SEM = load('SPECFEM3D_10_10.txt');
% time3 = SEM(:,1);
% vs1_3 = SEM(:,3);

figure
% plot(time, vs1, 'g', 'LineWidth', 1.5);
% hold on;
% plot(time2, vs1_2, 'b', 'LineWidth', 1.5);
% hold on;
% plot(time3, vs1_3, 'r', 'LineWidth', 1.5);
% hold on;
% plot(t, -v, 'k', 'LineWidth', 1.5);
plot(t, v, 'k', 'LineWidth', 1.5);
% hold on;
% legend('EQdyna', 'DayFD', 'SPECFEM3D', 'CGFDM');
xlabel('T (sec)');
xlim([0, 15]);
title(var);
set(gca, 'FontSize', 12);

