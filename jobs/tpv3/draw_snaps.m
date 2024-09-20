clc;
clear;
% close all
addpath("../../matlab/");
parfile = 'params.json';
par = get_params(parfile);
NY = par.NY;
NZ = par.NZ;
DT = par.DT;
TMAX = par.TMAX;
TSKP = par.EXPORT_TIME_SKIP;
nfault = par.num_fault;
Faultgrid = par.Fault_grid;
j1 = Faultgrid(1);
j2 = Faultgrid(2);
k1 = Faultgrid(3);
k2 = Faultgrid(4);
NT = floor(TMAX/(DT*TSKP));

its = 1:20:2000;
its = floor(its);
nt = length(its);

[x,y,z] = gather_coord(parfile);
x = x*1e-3;
y = y*1e-3;
z = z*1e-3;
figure;
filename1='Vs1.gif';
for i = 1:nt 
it = its(i);
disp(it);
out = par.OUT;
% Vs1 = gather_snap(parfile,out,'str_peak', nfault);
Vs1 = gather_snap(parfile,out,'ts1', it);
% Vs2 = gather_snap(parfile,out,'Vs2',it);

% pcolor(y, z, Vs1);
pcolor(y(j1:j2,k1:k2), z(j1:j2,k1:k2), Vs1(j1:j2,k1:k2)); 
axis equal; 
shading interp;
% view([60, 30]);    

%caxis([0 1]);
colormap( 'jet' );
colorbar;
title(['Strike-slip rate t=', num2str(it*DT*TSKP),'s'],'FontSize',12)
set(gca,'FontSize',12)
%axis([-1 1 -1 0]*15)
pause(0.01)
drawnow;
    % im=frame2im(getframe(gcf));
    % [imind,map]=rgb2ind(im,256);
%    if i==1
%         % ���Ѹ���ģʽд��ָ����gif�ļ�
%        imwrite(imind,map,filename1,'gif','LoopCount',Inf,'DelayTime',0.2);
%    else
%         % ����׷��ģʽ��ÿһ֡д��gif�ļ�
%        imwrite(imind,map,filename1,'gif','WriteMode','append','DelayTime',0.2);
%    end
end
