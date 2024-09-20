clc
clear
addpath('../../matlab/');
par = get_params('params.json');

ny = par.NY
nz = par.NZ
dh = par.DH
OUT = par.OUT

fnm = par.Fault_geometry
x = ncread(fnm, 'x');
y = ncread(fnm, 'y');
z = ncread(fnm, 'z');
vec_n = ncread(fnm, 'vec_n');
vec_m = ncread(fnm, 'vec_m');
vec_l = ncread(fnm, 'vec_l');

%x = imgaussfilt(x, [1 1]*10);
%x = x/max(max(abs(x)));
%x = x * 300;
%if flag_flat
%x(:,:) = 0;
%end

Tx = zeros(ny, nz);
Ty = zeros(ny, nz);
Tz = zeros(ny, nz);

Tn = zeros(ny, nz);
Tm = zeros(ny, nz);
Tl = zeros(ny, nz);

dTx = zeros(ny, nz);
dTy = zeros(ny, nz);
dTz = zeros(ny, nz);

dTn = zeros(ny, nz);
dTm = zeros(ny, nz);
dTl = zeros(ny, nz);

for k = 1:nz
  if(~mod(k,50)) disp([num2str(k), '/', num2str(nz)]); end
  for j = 1:ny
  
    T_global = [
    -120 -70 0;
    -70    0 0;
      0    0 0;
    ]*1e6;

    if(abs(y(j,k)-0) < 1.5e3 && abs(z(j,k)+7.5e3) < 1.5e3)
    T_global = [
    -120 -81.6 0;
    -81.6    0 0;
      0      0 0;
    ]*1e6;
    end
    
    en = squeeze(vec_n(:,j,k));
    em = squeeze(vec_m(:,j,k));
    el = squeeze(vec_l(:,j,k));
    %^if(flag_flat)
    %^en = [1 0 0]';
    %^em = [0 1 0]';
    %^el = [0 0 1]';
    %^end
    
    
    Txyz = T_global * en;
    Tx(j,k) = Txyz(1);
    Ty(j,k) = Txyz(2);
    Tz(j,k) = Txyz(3);
    
    % add nuke
    %tn = dot(Txyz, en);
    %ts_vec = Txyz-tn*en;
    %ts = norm(ts_vec);
    %
    %tau_nuke = 0;
    %r = sqrt((y(j,k)+0.0e3)^2+(z(j,k)+7.5e3)^2);
    %%if(r<1.4e3)
    %%  tau_nuke = 11.6e6;
    %%elseif(r<2.0e3)
    %%  tau_nuke = 5.8e6*(1+cos(pi*(r-1.4e3)/600));
    %%else
    %%  tau_nuke = 0;
    %%end
    %R = 3e3;
    %if(r<R)
    %  Fr = exp(r^2/(r^2-R^2));
    %else
    %  Fr = 0;
    %end
    %tau_nuke = 25e6*Fr;
    %ts_new = ts+tau_nuke;
    %ts = max(ts, 1);
    %ts_vec_new = ts_new/ts * ts_vec;
    %ts_vec_nuke = tau_nuke/ts * ts_vec;
    
    %Txyz = tn*en+ts_vec_new;
    %Txyz = 0*en+ts_vec_nuke;

    Txyz(:) = 0;
    
    dTx(j,k) = Txyz(1);
    dTy(j,k) = Txyz(2);
    dTz(j,k) = Txyz(3);
  
    Tn(j,k)=Tx(j,k)*en(1)+Ty(j,k)*en(2)+Tz(j,k)*en(3);
    Tm(j,k)=Tx(j,k)*em(1)+Ty(j,k)*em(2)+Tz(j,k)*em(3);
    Tl(j,k)=Tx(j,k)*el(1)+Ty(j,k)*el(2)+Tz(j,k)*el(3);

    dTn(j,k)=dTx(j,k)*en(1)+dTy(j,k)*en(2)+dTz(j,k)*en(3);
    dTm(j,k)=dTx(j,k)*em(1)+dTy(j,k)*em(2)+dTz(j,k)*em(3);
    dTl(j,k)=dTx(j,k)*el(1)+dTy(j,k)*el(2)+dTz(j,k)*el(3);
  end
end

%% slip weakening friction
mu_s = zeros(ny, nz);
mu_d = zeros(ny, nz);
Dc = zeros(ny, nz);
C0 = zeros(ny, nz);

j1 = par.Fault_grid(1);
j2 = par.Fault_grid(2);
k1 = par.Fault_grid(3);
k2 = par.Fault_grid(4);

mu_s(:,:) = 10000;
mu_d(:,:) = 0.525;
Dc(:,:) = 0.4;
C0(:,:) = 1000e6;

mu_s(j1:j2,k1:k2) = 0.677;
C0(j1:j2,k1:k2) = 0;
%% rate state friction
V0 = 1e-6;
Vini = 1e-12;

W = 15e3; w = 3e3;
By = Bfunc(y, W, w);
Bz = Bfunc(z+7.5e3, W/2, w);
B = (1-By.*Bz);
a = 0.008+0.008*B;
Vw = 0.1+0.9*B;
b = ones(size(a))*0.012;
L = ones(size(a))*0.02;

Tau = sqrt(Tm.^2+Tl.^2);
State = a.*log(2.0*V0/Vini*sinh(Tau./abs(Tn)./a));

fnm_out = par.Fault_init_stress
ncid = netcdf.create(fnm_out, 'CLOBBER');
dimid(1) = netcdf.defDim(ncid,'ny',ny);
dimid(2) = netcdf.defDim(ncid,'nz',nz);
varid(1) = netcdf.defVar(ncid,'x','NC_FLOAT',dimid);
varid(2) = netcdf.defVar(ncid,'y','NC_FLOAT',dimid);
varid(3) = netcdf.defVar(ncid,'z','NC_FLOAT',dimid);
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
varid(14) = netcdf.defVar(ncid,'a','NC_FLOAT',dimid);
varid(15) = netcdf.defVar(ncid,'b','NC_FLOAT',dimid);
varid(16) = netcdf.defVar(ncid,'L','NC_FLOAT',dimid);
varid(17) = netcdf.defVar(ncid,'Vw','NC_FLOAT',dimid);
varid(18) = netcdf.defVar(ncid,'State','NC_FLOAT',dimid);
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
netcdf.putVar(ncid,varid(14),a);
netcdf.putVar(ncid,varid(15),b);
netcdf.putVar(ncid,varid(16),L);
netcdf.putVar(ncid,varid(17),Vw);
netcdf.putVar(ncid,varid(18),State);
netcdf.close(ncid);
