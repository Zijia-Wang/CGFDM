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
j1 = par.Fault_grid(1:4:end-3);
j2 = par.Fault_grid(2:4:end-2);
k1 = par.Fault_grid(3:4:end-1);
k2 = par.Fault_grid(4:4:end);
NT = floor(TMAX/(DT*TSKP));

flag_image = 1

its = 10:10:NT;
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
% Vs1 = gather_snap(parfile,out,'str_peak');
% tn = gather_snap(parfile,out,'tn',it);
Vs1 = gather_snap(parfile,out,'Vs1',it);
for n = 1 : nfault
    x1 = squeeze(x(:,:, n));
    y1 = squeeze(y(:,:, n));
    z1 = squeeze(z(:,:, n));
    surf(x1(j1(n):j2(n),k1(n):k2(n)), y1(j1(n):j2(n),k1(n):k2(n)), z1(j1(n):j2(n),k1(n):k2(n)), squeeze(Vs1(j1(n):j2(n),k1(n):k2(n), n)));
    hold on;
end
hold off;
view([60, 30]); 
axis equal; 
shading interp;
% xlim([-15, -5]);
% ylim([-15, -5]);
%caxis([0 1]*vm)
colormap( 'jet' );
colorbar;
title(['Strike-slip rate t=', num2str(it*DT*TSKP),'s'],'FontSize',12)
set(gca,'FontSize',12)
%axis([-1 1 -1 0]*15)
pause(0.01)
drawnow;
%     im=frame2im(getframe(gcf));
%     [imind,map]=rgb2ind(im,256);
%    if i==1
%         % ���Ѹ���ģʽд��ָ����gif�ļ�
%        imwrite(imind,map,filename1,'gif','LoopCount',Inf,'DelayTime',0.2);
%    else
%         % ����׷��ģʽ��ÿһ֡д��gif�ļ�
%        imwrite(imind,map,filename1,'gif','WriteMode','append','DelayTime',0.2);
%    end
end
