clc
clear
addpath('../../matlab');
parfile = 'params.json';
par = get_params(parfile);
NY = par.NY;
NZ = par.NZ;
NF = par.num_fault;
DT = par.DT;
DH = par.DH;
TMAX = par.TMAX;
TSKP = par.EXPORT_TIME_SKIP;

j1 = par.Fault_grid(1:4:end-3);
j2 = par.Fault_grid(2:4:end-2);
k1 = par.Fault_grid(3:4:end-1);
k2 = par.Fault_grid(4:4:end);

NT = floor(TMAX/(TSKP*DT));

[x,y,z] = gather_coord(parfile);
x = x * 1e-3;
y = y * 1e-3;
z = z * 1e-3;

t = gather_snap(parfile,par.OUT,'init_t0');

for i = 1 : NF
    
    y1 = squeeze(y(:,:,i));
    z1 = squeeze(z(:,:,i));
    T = squeeze(t(:,:,i));
    v = cal_rup_v(T, DH);
    v = v / 3464;
    v(v<0) = nan;
    v(v>2) = nan;

    vec = 0.5:0.5:20;

    figure;
    pcolor(y1(j1(i):j2(i),k1(i):k2(i)),z1(j1(i):j2(i),k1(i):k2(i)),v(j1(i):j2(i),k1(i):k2(i)));
    shading interp;
    caxis([0.5 1.5]);
    hold on;
    contour(y1(j1(i):j2(i),k1(i):k2(i)), z1(j1(i):j2(i),k1(i):k2(i)), T(j1(i):j2(i),k1(i):k2(i)), vec, 'color', 'k', 'linewidth', 1.5);
    hold off;
    title(['Vr / Vs (Fault ' num2str(i) ')']);
    colorbar;
    axis image;axis xy;
    colormap( jet );
    ylabel('Down-dip (km)');
    xlabel('Along-strike (km)');

    set(gcf,'PaperPositionMode', 'auto')
%print('-depsc', '-painters', 'tpv102_rupture_time_rough')
end