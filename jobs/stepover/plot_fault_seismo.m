clc
clear
addpath('../../matlab/')
parfile = 'params.json';
par = get_params(parfile);
% outdir = par.OUT;
outdir = './compareE';
nF = [1 1 1 1 1 2 2 2 2];
% nF = [2 2 2 2 2 1 1 1 1];
yall = [10000 20000 30000 36000 38000 40000 40000 45000 38000];
zall = [-7500 -5000 0 -5000 -10000 0 -10000 -5000 -12000];

% yall = 40000;
% zall = -12000;
for i = 1:length(yall)
    y=yall(i);
    z=zall(i);
    nfault = nF(i);
    var = 'Vs1';
[v1, t] = get_fault_seismo(parfile,outdir,var,y,z, nfault);
var = 'ts1';
[v2, ~] = get_fault_seismo(parfile,outdir,var,y,z, nfault);
var = 'State';
[v3, ~] = get_fault_seismo(parfile,outdir,var,y,z, nfault);
var = 'tn';
[v4, ~] = get_fault_seismo(parfile,outdir,var,y,z, nfault);

figure;
subplot(2,2,1);
plot(t, -v1);
title('Vs1');
subplot(2,2,2);
plot(t, -v2);
title('ts1');
subplot(2,2,3);
plot(t, v3);
title('State');
subplot(2,2,4);
title('tn');
plot(t, v4);
filename = ['./compareE/' num2str(y/1e3) '_' num2str(-z/1e3) '.txt']
data = [t, v1, v2, v3, v4];
save(filename,'data','-ASCII');
xlabel('T (sec');
end
%{
nfault = 2;
y = 39500
z = -5000
var = 'tn'
[v1, t] = get_fault_seismo(parfile,outdir,var,y,z, nfault);
var = 'ts1'
[vt1, ~] = get_fault_seismo(parfile,outdir,var,y,z, nfault);
var = 'ts2'
[vt2, ~] = get_fault_seismo(parfile,outdir,var,y,z, nfault);
v2 = sqrt(vt1.^2 + vt2.^2);
var = 'State'
[v3, ~] = get_fault_seismo(parfile,outdir,var,y,z, nfault);
var = 'Vs1'
[vs1, ~] = get_fault_seismo(parfile,outdir,var,y,z, nfault);
var = 'Vs2'
[vs2, ~] = get_fault_seismo(parfile,outdir,var,y,z, nfault);
v4 = sqrt(vs1.^2 + vs2.^2);
% v4 = log10(v4);
v4(v4<1e-16) = 1e-16;

% outdir = 'outC0.48';
% var = 'tn'
% [v1h, t] = get_fault_seismo(parfile,outdir,var,y,z, nfault);
% var = 'ts1'
% [vt1, ~] = get_fault_seismo(parfile,outdir,var,y,z, nfault);
% var = 'ts2'
% [vt2, ~] = get_fault_seismo(parfile,outdir,var,y,z, nfault);
% v2h = sqrt(vt1.^2 + vt2.^2);
% var = 'State'
% [v3h, ~] = get_fault_seismo(parfile,outdir,var,y,z, nfault);
% var = 'Vs1'
% [vs1, ~] = get_fault_seismo(parfile,outdir,var,y,z, nfault);
% var = 'Vs2'
% [vs2, ~] = get_fault_seismo(parfile,outdir,var,y,z, nfault);
% v4h = sqrt(vs1.^2 + vs2.^2);
% % v4h = log10(v4h);
% v4h(v4h<1e-16) = 1e-16;

sta_num = 1;
TMAX = par.TMAX;
DT = par.DT;
TSKIP = par.EXPORT_TIME_SKIP;
NT = floor(TMAX/(DT*TSKIP))-1;
sta_x = [1350];
sta_y = [39500];

stations = zeros(sta_num, 2);
stations(:, 1) = sta_x;
stations(:, 2) = sta_y;

dirname = 'outE0.45';
[vx, vy, vz] = read_wave(parfile, dirname, stations);
t2 = (0:NT)*DT*TSKIP;
% dirname = 'outC0.48';
% [vx2, vy2, vz2] = read_wave(parfile, dirname, stations);

% var = 'Vs2'
% [v5, ~] = get_fault_seismo(parfile,outdir,var,y,z, nfault);
% rate = sqrt(v4.^2+v5.^2);
% a = 0.01;
% mu = a*asinh(rate.*exp(v3/0.01)/(2*1e-6));figure;
figure;
subplot(2,2,1);
yyaxis left;
plot(t, v1/1e6, 'k', 'LineWidth', 1.5);
hold on;
% plot(t, v1h/1e6, 'r-', 'LineWidth', 1.5);
% hold on;
xlim([8, 14]);
% title('Normal stress');
ylabel('Normal stress (MPa)');
xlabel('T (sec)');
yyaxis right;
plot(t2, vx, 'k--', 'LineWidth', 1.5);
hold on;
% plot(t2, vx2,'r--', 'LineWidth', 1.5);
% ylabel('Vx (m/s)');
legend('\sigma_n', 'Vx');
% legend('\sigma_n, Te = 0.45', '\sigma_n, Te = 0.48','Vx, Te = 0.45', 'Vx, Te = 0.48');
% xlim([10, 14]);
set(gca, 'FontSize', 12);
subplot(2,2,2);
plot(t, v2/1e6, 'k', 'LineWidth', 1.5);
hold on;
% plot(t, v2h/1e6, 'r', 'LineWidth', 1.5);

% legend('Te = 0.45', 'Te = 0.48');
% xlim([0, 18]);
% title('Shear stress');
% xlabel('T (sec)');
% ylabel('MPa');
% xlim([10, 14]);
hold on;
xlim([8, 14]);
% title('Normal stress');
ylabel('Shear stress (MPa)');
xlabel('T (sec)');
yyaxis right;
plot(t2, vy, 'k--', 'LineWidth', 1.5);
% hold on;
% plot(t2, vy2,'r--', 'LineWidth', 1.5);
ylabel('Vy (m/s)');
legend('\tau', 'Vy');
% legend('\tau, Te = 0.45', '\tau, Te = 0.48','Vy, Te = 0.45', 'Vy, Te = 0.48');
set(gca, 'FontSize', 12);

subplot(2,2,3);
% plot(t, v3, 'k', 'LineWidth', 1.5);
% hold on;
% plot(t, v3h, 'r', 'LineWidth', 1.5);
% legend('Te = 0.45', 'Te = 0.48');
% xlim([0, 18]);
% title('State');
% xlabel('T (sec)');
% xlim([10, 14]);
plot(t2, vz, 'k', 'LineWidth', 1.5);
hold on;
% plot(t2, vz2,'r', 'LineWidth', 1.5);
% legend('Te = 0.45', 'Te = 0.48');
xlabel('T (sec)');
ylabel('m/s')
xlim([8, 14]);
title('Vz');
set(gca, 'FontSize', 12);

subplot(2,2,4);
semilogy(t, v4, 'k', 'LineWidth', 1.5);
% hold on;
% semilogy(t, v4h, 'r', 'LineWidth', 1.5);
% legend('Te = 0.45', 'Te = 0.48');
% xlim([0, 18]);
xlim([8, 14]);
ylim([1e-16, 15]);
title('Slip rate');
xlabel('T (sec)');
ylabel('m/s');
% xlim([10, 14]);
set(gca, 'FontSize', 12);
%}
