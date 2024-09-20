clc
clear
addpath('../../matlab');
par = get_params('params.json');
nx = par.NX;
ny = par.NY;
nz = par.NZ;
dh = par.DH;
OUT = par.OUT;
srci = par.src_i;
num_fault = par.num_fault;
f0 = par.RS_f0;
fw = par.RS_fw;
Vini = par.RS_Vini;
V0 = par.RS_V0;

nuc_seg = 1;
j0 = 10000;
k0 = -7500;
R0 = 2000;
R = 2000;
tn = -80*1e6;
Te = 0.45;
tau = (Te*f0 + (1-Te)*fw) * tn
Te = (abs(tau)-fw*abs(tn)) / ((f0-fw)*abs(tn))

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

F = zeros(ny, nz);

for i = 1 : num_fault
    ix = srci(i);
%     x1 = squeeze(x(:,:, ix));
    y1 = squeeze(y(:,:, ix));
    z1 = squeeze(z(:,:, ix));
    for k = 1:nz
        if(~mod(k,50)) disp([num2str(k), '/', num2str(nz)]); end
%         depth = -z1(1, k);
        depth = abs(z1(1,k));
%         if(depth < 1e-3)
%             depth = dh/3;
%         end
%         if depth < 5.0e3
%             tn = tn * max(depth, dh/3.0)/5.e3;
%             tau = tau * max(depth, dh/3.0)/5.e3;
%         end
        for j = 1:ny

            en = squeeze(vec_n(i,:,j,k));
            em = squeeze(vec_m(i,:,j,k));
            el = squeeze(vec_l(i,:,j,k));
%             Tn(j,k,i) = -120*1e6;
            Tn(j,k,i) = tn;
%             if depth < 5.0e3
%                Tn(j,k,i) = Tn(j,k,i)*max(depth, dh/3.0)/5.e3;
%             end
%             if (k == nz) 
%                 Tn(j,k,i) = -7378.0*dh/3.0; 
%             end
            Tl(j,k,i) = 0.0;
%             Tm(j,k,i) = -75*1e6;
            Tm(j,k,i) = tau;

            Tx(j,k,i)=Tn(j,k,i)*en(1)+Tm(j,k,i)*em(1)+Tl(j,k,i)*el(1);
            Ty(j,k,i)=Tn(j,k,i)*en(2)+Tm(j,k,i)*em(2)+Tl(j,k,i)*el(2);
            Tz(j,k,i)=Tn(j,k,i)*en(3)+Tm(j,k,i)*em(3)+Tl(j,k,i)*el(3);

            Txyz = [Tx(j, k, i), Ty(j, k, i), Tz(j, k, i)];

            r = sqrt((y1(j,k)-j0)^2+(z1(j,k)-k0)^2);
   
%     R = 1400;
%     R = 2000;
    if(r<R0 && i == nuc_seg)
      Fr = exp(r^2/(r^2-R0^2));
%       F(j,k) = Fr;
    else
      Fr = 0;
    end
%     tau_nuke = 100e6*Fr;
    tau_nuke = (0.87*abs(tn)-abs(tau))*Fr; %0.86
    tn = dot(Txyz, en);
    ts_vec = Txyz-tn*en;
    ts = norm(ts_vec);
    ts = max(ts, 1);
    ts_vec_nuke = tau_nuke/ts * ts_vec;

    Txyz = 0*en+ts_vec_nuke;
    dTx(j,k,i) = Txyz(1);
    dTy(j,k,i) = Txyz(2);
    dTz(j,k,i) = Txyz(3);

            Tn(j, k, i)=Tx(j, k, i)*en(1)+Ty(j, k, i)*en(2)+Tz(j, k, i)*en(3);
            Tm(j, k, i)=Tx(j, k, i)*em(1)+Ty(j, k, i)*em(2)+Tz(j, k, i)*em(3);
            Tl(j, k, i)=Tx(j, k, i)*el(1)+Ty(j, k, i)*el(2)+Tz(j, k, i)*el(3);

        end
    end
end
if R<R0
Tnuc0 = sum(sum(dTy(:,:,nuc_seg)/1e6))

ix = srci(nuc_seg);
y1 = squeeze(y(:,:, ix));
z1 = squeeze(z(:,:, ix));
r = sqrt((y1-j0).^2+(z1-k0).^2);

F(r<R) = exp(r(r<R).^2./(r(r<R).^2-R^2));

Tnuc1 = abs(Tnuc0) / sum(sum(F))
tau_nuke = Tnuc1*F*1e6; %0.86
dTx(:,:,:) = 0;
dTy(:,:,:) = 0;
dTz(:,:,:) = 0;
for k = 1:nz
    if(~mod(k,50)) disp([num2str(k), '/', num2str(nz)]); end
    for j = 1:ny
        en = squeeze(vec_n(nuc_seg,:,j,k));
        Txyz = [Tx(j, k, nuc_seg), Ty(j, k, nuc_seg), Tz(j, k, nuc_seg)];
        tn = dot(Txyz, en);
        ts_vec = Txyz-tn*en;
        ts = norm(ts_vec);
        ts = max(ts, 1);
        ts_vec_nuke = tau_nuke(j,k)/ts * ts_vec;

        Txyz = 0*en+ts_vec_nuke;
        dTx(j,k,nuc_seg) = Txyz(1);
        dTy(j,k,nuc_seg) = Txyz(2);
        dTz(j,k,nuc_seg) = Txyz(3);
    end
end
Tnuc2 = sum(sum(dTy(:,:,nuc_seg)))
Tnuc2 - Tnuc0*1e6
end
%% slip weakening friction
% mu_s = zeros(ny, nz, num_fault);
% mu_d = zeros(ny, nz, num_fault);
% Dc = zeros(ny, nz, num_fault);
% C0 = zeros(ny, nz, num_fault);
% 
j1 = par.Fault_grid(1:4:end-3);
j2 = par.Fault_grid(2:4:end-2);
k1 = par.Fault_grid(3:4:end-1);
k2 = par.Fault_grid(4:4:end);
% 
% % j3 = par.Fault_grid(1);
% % j4 = par.Fault_grid(2);
% % k3 = par.Fault_grid(3);
% % k4 = par.Fault_grid(4);
% 
% mu_s(:,:,:) = 10000;
% mu_d(:,:,:) = 0.448;
% Dc(:,:,:) = 0.5;
% C0(:,:,:) = 1000e6;
% % for k = 1 : nz
% %     depth = (nz-k)*dh;
% %     if depth < 5000
% %         C0(:,k,:) = 0.0014*(5000-depth)*1e6;
% %     end
% % end
% for i = 2 : num_fault
%     mu_s(j1(i):j2(i),k1(i):k2(i), i) = 0.76;
%     C0(j1(i):j2(i),k1(i):k2(i), i) = 0.2e6;
% end
% mu_s(j3:j4,k3:k4, 2) = 0.548;
% C0(j1:j2,k1:k2) = 0;
%% rate state friction
b = zeros(ny, nz, num_fault);
a = zeros(ny, nz, num_fault);
Vw = zeros(ny, nz, num_fault);
L = zeros(ny, nz, num_fault);
State = zeros(ny, nz, num_fault);
B = ones(ny, nz, num_fault);
% 
for i = 1 : num_fault
    ix = srci(i);
%     x1 = squeeze(x(:,:, ix));
    y1 = squeeze(y(:,:, ix));
    z1 = squeeze(z(:,:, ix));
    
    if i == 1
        W = 25000; w = 0;
        
    else
        W = 25000; w = 0;
    end
    By = Bfunc(y1-25000, W, w);
    Bz = Bfunc(z1+7.5e3, 7500, 0);
    B = (1-By.*Bz);
    if i == nuc_seg
        B(j2(i)+1:end, :) = 1;
    else
        B(1:j1(i)-1, :) = 1;
    end
    
    a(:,:,i) = 0.01+0.01*B;%(:,:,i);
    Vw(:,:,i) = 0.1+0.9*B;%(:,:,i);
    b(:,:,i) = 0.014;
%     L(:,:,i) = 0.02015;  % AL
    L(:,:,i) = 0.2;        % SL
%     b = ones(size(a))*0.012;
%     L = ones(size(a))*0.02;

    Tau = sqrt(Tm(:,:,i).^2+Tl(:,:,i).^2);
    State(:,:,i) = a(:,:,i).*log(2.0*V0/Vini*sinh(Tau./abs(Tn(:,:,i))./a(:,:,i)));
end

% 
% Tau = sqrt(Tm.^2+Tl.^2);
% State = a.*log(2.0*V0/Vini*sinh(Tau./abs(Tn)./a));

fnm_out = par.Fault_init_stress
ncid = netcdf.create(fnm_out, 'NETCDF4');

dimid(1) = netcdf.defDim(ncid,'ny',ny);
dimid(2) = netcdf.defDim(ncid,'nz',nz);
dimid(3) = netcdf.defDim(ncid,'num_fault',num_fault);

dimidxyz(1) = dimid(1);
dimidxyz(2) = dimid(2);
dimidxyz(3) = netcdf.defDim(ncid, 'nx', nx);
varid(1) = netcdf.defVar(ncid,'x','NC_DOUBLE',dimidxyz);
varid(2) = netcdf.defVar(ncid,'y','NC_DOUBLE',dimidxyz);
varid(3) = netcdf.defVar(ncid,'z','NC_DOUBLE',dimidxyz);
varid(4) = netcdf.defVar(ncid,'Tx','NC_DOUBLE',dimid);
varid(5) = netcdf.defVar(ncid,'Ty','NC_DOUBLE',dimid);
varid(6) = netcdf.defVar(ncid,'Tz','NC_DOUBLE',dimid);
varid(7) = netcdf.defVar(ncid,'dTx','NC_DOUBLE',dimid);
varid(8) = netcdf.defVar(ncid,'dTy','NC_DOUBLE',dimid);
varid(9) = netcdf.defVar(ncid,'dTz','NC_DOUBLE',dimid);
varid(10) = netcdf.defVar(ncid,'a','NC_DOUBLE',dimid);
varid(11) = netcdf.defVar(ncid,'b','NC_DOUBLE',dimid);
varid(12) = netcdf.defVar(ncid,'L','NC_DOUBLE',dimid);
varid(13) = netcdf.defVar(ncid,'Vw','NC_DOUBLE',dimid);
varid(14) = netcdf.defVar(ncid,'State','NC_DOUBLE',dimid);
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
netcdf.putVar(ncid,varid(10),a);
netcdf.putVar(ncid,varid(11),b);
netcdf.putVar(ncid,varid(12),L);
netcdf.putVar(ncid,varid(13),Vw);
netcdf.putVar(ncid,varid(14),State);
netcdf.close(ncid);

figure;
for i = 1 : num_fault
    ix = srci(i);
    x1 = squeeze(x(:,:, ix));
    y1 = squeeze(y(:,:, ix));
    z1 = squeeze(z(:,:, ix));

    surf(x1(j1(i):j2(i),k1(i):k2(i)), y1(j1(i):j2(i),k1(i):k2(i)), z1(j1(i):j2(i),k1(i):k2(i)), squeeze(dTy(j1(i):j2(i),k1(i):k2(i), i)));
    hold on;
    view([-60,30]);
    shading interp;
    colorbar;
    axis image;
end
