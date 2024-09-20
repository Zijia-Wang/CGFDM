clc
clear
close all

parfile = 'params.json';
par = get_params(parfile);
% NX = par.NX;
NY = par.NY;
NZ = par.NZ;
DT = par.DT;
TMAX = par.TMAX;
TSKP = par.EXPORT_TIME_SKIP;

NT = floor(TMAX/(DT*TSKP));

its = 2000:10:3500;
its = floor(its);
nt = length(its);

[x,y,z] = gather_coord_xy(parfile);
x = x*1e-3;
y = y*1e-3;
z = z*1e-3;

for i = 1:nt 
it = its(i);
disp(it);
out = par.OUT;
Vx = gather_snap_wave_xy(parfile,out,'Vx',it);
Vy = gather_snap_wave_xy(parfile,out,'Vy',it);
% Vz = gather_snap_wave_xy(parfile,out,'Vz',it);

% v = sqrt(Vx.^2+Vy.^2+Vz.^2);

pcolor(x, y, Vy);
shading interp

axis image;axis xy;
% vm = max(max(abs(v)))/2;
% caxis([-2 2])
colormap( jet );
colorbar;
title([num2str(it*DT*TSKP),'s'],'FontSize',12);
set(gca,'FontSize',12);
%axis([-1 1 -1 0]*15)
pause(0.01);
end

