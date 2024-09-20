clc
clear
% close all

parfile = 'params.json';
par = get_params(parfile);
% NX = par.NX;
NY = par.NY;
NZ = par.NZ;
DT = par.DT;
TMAX = par.TMAX;
TSKP = par.EXPORT_TIME_SKIP;

NT = floor(TMAX/(DT*TSKP));

its = 20:2:NT;
its = floor(its)
nt = length(its);

[x,y,z] = gather_coord_xy(parfile);
x = x*1e-3;
y = y*1e-3;
z = z*1e-3;
figure;
for i = 1:nt 
it = its(i);
disp(it);
out = par.OUT;
%Vs1 = gather_snap(parfile,out,'Vs1',it);
%Vs2 = gather_snap(parfile,out,'Vs2',it);
Vx = gather_snap_wave_xy(parfile,out,'Vx',it);
Vy = gather_snap_wave_xy(parfile,out,'Vy',it);
% Vz = gather_snap_wave_xy(parfile,out,'Vz',it);

% v = sqrt(Vx.^2+Vy.^2+Vz.^2);

pcolor(x, y, Vy);
shading interp

axis image;axis xy
% vm = max(max(abs(v)))/2;
% caxis([-5 5])
colormap( jet )
colorbar
title([num2str(it*DT*TSKP),'s'],'FontSize',12)
set(gca,'FontSize',12)
%axis([-1 1 -1 0]*15)
pause(0.01)
drawnow;
end

%subplot(5,3,15)
%axes('position',[0.2,0.02,.6,.15])
%axis off
%colorbar('south','position',[0.44,0.04,0.15,0.015]);
%caxis([0 5]);
%colorbar('horiz')
%set(gca,'LooseInset',get(gca,'TightInset'))
%set(gca,'looseInset',[0 0 0 0])
%set(gcf, 'PaperPositionMode', 'auto')
%picname = 'tpv102_rough300m'
%picname = 'tpv102_flat'
%if flag_image
%print('-depsc', '-r300', picname)
%else
%print('-dpng', '-r300', picname)
%end
