clear;
addpath('../../matlab/');
parfile = 'params.json';
par = get_params(parfile);
dirname = par.OUT;
TSKIP = par.EXPORT_TIME_SKIP;
DT = par.DT;
TMAX = par.TMAX;
NT = floor(TMAX/(DT*TSKIP))-1;
sta_num = 1;

sta_x = [3000];
sta_y = [15000];

stations = zeros(sta_num, 2);
stations(:, 1) = sta_x;
stations(:, 2) = sta_y;

[vx, vy, vz] = read_wave(parfile, dirname, stations);
t = (0:NT-1)*DT*TSKIP;

% EQDyna = load('EQdyna__3_15.txt');
% time = EQDyna(:,1);
% vx1 = EQDyna(:,7);
% vy1 = EQDyna(:,3);
% vz1 = EQDyna(:,5);
% 
% DayFD = load('DayFD__3_15.txt');
% time2 = DayFD(:,1);
% vx2 = DayFD(:,7);
% vy2 = DayFD(:,3);
% vz2 = DayFD(:,5);
% 
% SEM = load('SPECFEM3D__3_15.txt');
% time3 = SEM(:,1);
% vx3 = SEM(:,7);
% vy3 = SEM(:,3);
% vz3 = SEM(:,5);

figure;
subplot(1,3,1);
% plot(time, vx1, 'g', 'LineWidth', 1.5);
% hold on;
% plot(time2, vx2, 'b', 'LineWidth', 1.5);
% hold on;
% plot(time3, vx3, 'r', 'LineWidth', 1.5);
% hold on;
% plot(t, -vx, 'k', 'LineWidth', 1.5);
plot(t, vx, 'k', 'LineWidth', 1.5);
% hold on;
% legend('EQdyna', 'DayFD', 'SPECFEM3D', 'CGFDM');
xlabel('T (sec)');
xlim([0, 15]);
title('Vx');
set(gca, 'FontSize', 12);

subplot(1,3,2);
% plot(time, vy1, 'g', 'LineWidth', 1.5);
% hold on;
% plot(time2, vy2, 'b', 'LineWidth', 1.5);
% hold on;
% plot(time3, vy3, 'r', 'LineWidth', 1.5);
% hold on;
plot(t, vy, 'k', 'LineWidth', 1.5);
% hold on;
% legend('EQdyna', 'DayFD', 'SPECFEM3D', 'CGFDM');
xlabel('T (sec)');
xlim([0, 15]);
title('Vy');
set(gca, 'FontSize', 12);

subplot(1,3,3);
% plot(time, vz1, 'g', 'LineWidth', 1.5);
% hold on;
% plot(time2, vz2, 'b', 'LineWidth', 1.5);
% hold on;
% plot(time3, vz3, 'r', 'LineWidth', 1.5);
% hold on;
% plot(t, -vz, 'k', 'LineWidth', 1.5);
plot(t, vz, 'k', 'LineWidth', 1.5);
% hold on;
% legend('EQdyna', 'DayFD', 'SPECFEM3D', 'CGFDM');
xlabel('T (sec)');
xlim([0, 15]);
title('Vz');
set(gca, 'FontSize', 12);
