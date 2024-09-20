clc
clear

par = get_params('params.json');
nx = par.NX;
ny = par.NY;
nz = par.NZ;
dh = par.DH;
OUT = par.OUT;
srci = par.src_i;
num_fault = par.num_fault;

R1=1500;
R2=2000;

fnm = par.Fault_geometry;
x = ncread(fnm, 'x');
y = ncread(fnm, 'y');
z = ncread(fnm, 'z');
vec_n = ncread(fnm, 'vec_n');
vec_m = ncread(fnm, 'vec_s1');
vec_l = ncread(fnm, 'vec_s2');

%x = imgaussfilt(x, [1 1]*10);
%x = x/max(max(abs(x)));
%x = x * 300;
%if flag_flat
%x(:,:) = 0;
%end

Tx = zeros(ny, nz, num_fault);
Ty = zeros(ny, nz, num_fault);
Tz = zeros(ny, nz, num_fault);

Tn = zeros(ny, nz, num_fault);
Tm = zeros(ny, nz, num_fault);
Tl = zeros(ny, nz, num_fault);

dTx = zeros(ny, nz, num_fault);
dTy = zeros(ny, nz, num_fault);
dTz = zeros(ny, nz, num_fault);

for i = 1 : num_fault
    ix = srci(i);
    x1 = squeeze(x(:,:, ix));
    y1 = squeeze(y(:,:, ix));
    z1 = squeeze(z(:,:, ix));
    for k = 1:nz
        if(~mod(k,50)) disp([num2str(k), '/', num2str(nz)]); end
        depth = -z1(1, k);
        for j = 1:ny

            en = squeeze(vec_n(i,:,j,k));
            em = squeeze(vec_m(i,:,j,k));
            el = squeeze(vec_l(i,:,j,k));  
  
            Tn(j, k, i) = -60*1e6;
            if depth <= 15000
                Tm(j, k, i) = -29.38*1e6;
            else
                Tm(j, k, i) = (29.38 - (0.002938)*(depth - 15000))*(-1e6);
            end
            
            ts = Tm(j, k, i);
            ts_vec = ts * em;
            tn = Tn(j, k, i) * en;
            Txyz = ts_vec + tn;

            Tx(j, k, i) = Txyz(1);
            Ty(j, k, i) = Txyz(2);
            Tz(j, k, i) = Txyz(3);
  
            Tn(j, k, i)=Tx(j, k, i)*en(1)+Ty(j, k, i)*en(2)+Tz(j, k, i)*en(3);
            Tm(j, k, i)=Tx(j, k, i)*em(1)+Ty(j, k, i)*em(2)+Tz(j, k, i)*em(3);
            Tl(j, k, i)=Tx(j, k, i)*el(1)+Ty(j, k, i)*el(2)+Tz(j, k, i)*el(3);

        end
    end
end

%% slip weakening friction
mu_s = zeros(ny, nz, num_fault);
mu_d = zeros(ny, nz, num_fault);
Dc = zeros(ny, nz, num_fault);
C0 = zeros(ny, nz, num_fault);

j1 = par.Fault_grid(1:4:end-3);
j2 = par.Fault_grid(2:4:end-2);
k1 = par.Fault_grid(3:4:end-1);
k2 = par.Fault_grid(4:4:end);

mu_s(:,:,:) = 1000;
mu_d(:,:,:) = 0.373;
Dc(:,:,:) = 0.3;
for k = 1 : nz
    depth = (nz-k)*dh;
    if depth < 5000
        C0(:,k,:) = 0.0014*(5000-depth)*1e6;
    end
end

mu_s(j1(1):j2(1),k1(1):k2(1), 1) = 0.548;
mu_s(j1(2):j2(2),k1(1):k2(1), 2) = 0.548;
% C0(j1:j2,k1:k2) = 0;
%% rate state friction
% V0 = 1e-6;
% Vini = 1e-12;
% 
% W = 15e3; w = 3e3;
% By = Bfunc(y, W, w);
% Bz = Bfunc(z+7.5e3, W/2, w);
% B = (1-By.*Bz);
% a = 0.008+0.008*B;
% Vw = 0.1+0.9*B;
% b = ones(size(a))*0.012;
% L = ones(size(a))*0.02;
% 
% Tau = sqrt(Tm.^2+Tl.^2);
% State = a.*log(2.0*V0/Vini*sinh(Tau./abs(Tn)./a));

fnm_out = par.Fault_init_stress
ncid = netcdf.create(fnm_out, 'CLOBBER');

dimid(1) = netcdf.defDim(ncid,'ny',ny);
dimid(2) = netcdf.defDim(ncid,'nz',nz);
dimid(3) = netcdf.defDim(ncid,'num_fault',num_fault);

dimidxyz(1) = dimid(1);
dimidxyz(2) = dimid(2);
dimidxyz(3) = netcdf.defDim(ncid, 'nx', nx);
varid(1) = netcdf.defVar(ncid,'x','NC_FLOAT',dimidxyz);
varid(2) = netcdf.defVar(ncid,'y','NC_FLOAT',dimidxyz);
varid(3) = netcdf.defVar(ncid,'z','NC_FLOAT',dimidxyz);
varid(4) = netcdf.defVar(ncid,'Tx','NC_FLOAT',dimid);
varid(5) = netcdf.defVar(ncid,'Ty','NC_FLOAT',dimid);
varid(6) = netcdf.defVar(ncid,'Tz','NC_FLOAT',dimid);
varid(7) = netcdf.defVar(ncid,'dTx','NC_FLOAT',dimid);
varid(8) = netcdf.defVar(ncid,'dTy','NC_FLOAT',dimid);
varid(9) = netcdf.defVar(ncid,'dTz','NC_FLOAT',dimid);
varid(10) = netcdf.defVar(ncid,'mu_s','NC_FLOAT',dimid);
varid(11) = netcdf.defVar(ncid,'mu_d','NC_FLOAT',dimid);
varid(12) = netcdf.defVar(ncid,'Dc','NC_FLOAT',dimid);
varid(13) = netcdf.defVar(ncid,'C0','NC_FLOAT',dimid);
netcdf.endDef(ncid);
netcdf.putVar(ncid,varid(1),x);
netcdf.putVar(ncid,varid(2),y);
netcdf.putVar(ncid,varid(3),z);
netcdf.putVar(ncid,varid(4),Tx);
netcdf.putVar(ncid,varid(5),Ty);
netcdf.putVar(ncid,varid(6),Tz);
netcdf.putVar(ncid,varid(7),dTx);
netcdf.putVar(ncid,varid(8),dTy);
netcdf.putVar(ncid,varid(9),dTz);
netcdf.putVar(ncid,varid(10),mu_s);
netcdf.putVar(ncid,varid(11),mu_d);
netcdf.putVar(ncid,varid(12),Dc);
netcdf.putVar(ncid,varid(13),C0);
netcdf.close(ncid);

figure;
for i = 1 : num_fault
    ix = srci(i);
    x1 = squeeze(x(:,:, ix));
    y1 = squeeze(y(:,:, ix));
    z1 = squeeze(z(:,:, ix));
    surf(x1(j1(i):j2(i),k1(i):k2(i)), y1(j1(i):j2(i),k1(i):k2(i)), z1(j1(i):j2(i),k1(i):k2(i)), squeeze(Tm(j1(i):j2(i),k1(i):k2(i), i)));
    hold on;
    view([60,30]);
    shading interp;
    colorbar;
    axis image;
end
