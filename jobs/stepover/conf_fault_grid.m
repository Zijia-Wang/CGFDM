clc
clear
addpath('../../matlab/');
par = get_params('params.json');

nx = par.NX
ny = par.NY
nz = par.NZ
dh = par.DH
OUT = par.OUT
nfault = par.num_fault;
srci = par.src_i;
x1 = zeros(ny, nz);
y1 = zeros(ny, nz);
z1 = zeros(ny, nz);

for j = 1:ny
    for k = 1:nz
        x1(j,k) = 0;
%         y1(j,k) = (j-1-ny/2)*dh;
        y1(j,k) = (j-50)*dh;
        z1(j,k) = (k-nz)*dh;
    end
end

x = zeros(ny, nz, nx);
y = zeros(ny, nz, nx);
z = zeros(ny, nz, nx);

for i = 1 : nx
    x(:, :, i) = x1 + (i - srci(1)) * dh;
    y(:, :, i) = y1;
    z(:, :, i) = z1;
end

width1 = 10.0;
width2 = 55.0;
x(:, :, srci(1)) = x1;
stepwidth = 1400;   % 1700 61 1800 64  4000 112
dnx = srci(2) - srci(1);

dx = (stepwidth - x1) / dnx;

if stepwidth <= 1600
    for i = srci(1) + 1 : srci(2)
        dist = i - srci(1);
        x(:, :, i) = x(:, :, i - 1) + dh / 2;
    end
else
    x(:, :, srci(2)) = stepwidth;
    idc = floor(dnx / 2) + srci(1);
    for i = srci(1) + 1 : idc
        dist1 = i - srci(1);
%     dist2 = srci(2) - i;
        if (dist1 < width1) %|| (dist2 < width1)
              compr = 0;
        elseif (dist1 < width2)
              compr = 1.0 - cos(pi * (i - (srci(1) + width1)) / (width2 - width1));
%     elseif (dist2 < width2)
%         compr = 1.0 - cos(pi * (i - (srci(2) - width1)) / (width2 - width1));
        else
              compr = 2.0;
        end
        compr = 0.5 + 0.25 * compr;
        x(:, :, i) = x(:, :, i - 1) + dh * compr;
%     x(:, :, i) = x(:, :, i - 1) + dh / 2;
    end
    for i = srci(2) - 1 : -1 : idc+1
        dist = srci(2) - i;
        if (dist < width1)
              compr = 0;
        elseif (dist < width2)
              compr = 1.0 - cos(pi * (i - (srci(2) - width1)) / (width2 - width1));
        else
              compr = 2.0;
        end
        compr = 0.5 + 0.25 * compr;
        x(:, :, i) = x(:, :, i + 1) - dh * compr;
%     x(:, :, i) = x(:, :, i - 1) + dh / 2;
    end
end

for i = srci(2) + 1 : nx
	dist = i - srci(2);
    if (dist < width1)
		compr = 0;
    elseif (dist < width2)
		compr = 1.0 - cos(pi * (i - (srci(2) + width1)) / (width2 - width1));
    else
		compr = 2.0;
    end
		compr = 0.5 + 0.25 * compr;
		x(:, :, i) = x(:, :, i - 1) + dh * compr;
end
for i = srci(1) - 1 : -1 : 1
	dist = srci(1) - i;
    if (dist < width1)
		compr = 0;
    elseif (dist < width2)
		compr = 1.0 - cos(pi * (i - (srci(1) - width1)) / (width2 - width1));
    else
		compr = 2.0;
    end
        compr = 0.5 + 0.25 * compr;
        x(:, :, i) = x(:, :, i + 1) - dh * compr;
end

if 0
figure;
surf(x1, y1, z1);
hold on;
x1 = squeeze(x(:,:,120));
surf(x1, y1, z1);
shading flat
colorbar
axis image
end
figure;
X = squeeze(x(1,1,:));
plot(X)
disp('calculating metric and base vectors...')
vec_n = zeros(nfault, 3, ny, nz);
vec_m = zeros(nfault, 3, ny, nz);
vec_l = zeros(nfault, 3, ny, nz);
for n = 1: nfault
    ix = srci(n);
    x1 = squeeze(x(:, :, ix));
    metric = cal_metric(x1,y1,z1, dh);
    [v_n, v_m, v_l] = cal_basevectors(metric);
    vec_n(n, :,:,:) = v_n;
    vec_m(n, :,:,:) = v_m;
    vec_l(n, :,:,:) = v_l;
end

disp('write output...')
fnm_out = par.Fault_geometry
ncid = netcdf.create(fnm_out, 'NETCDF4');
dimid(1) = netcdf.defDim(ncid,'ny',ny);
dimid(2) = netcdf.defDim(ncid,'nz',nz);
dimid(3) = netcdf.defDim(ncid,'nx',nx);
dimid3(1) = netcdf.defDim(ncid, 'num_fault', nfault);
dimid3(2) = netcdf.defDim(ncid, 'dim', 3);
dimid3(3) = dimid(1);
dimid3(4) = dimid(2);
varid(1) = netcdf.defVar(ncid,'x','NC_FLOAT',dimid);
varid(2) = netcdf.defVar(ncid,'y','NC_FLOAT',dimid);
varid(3) = netcdf.defVar(ncid,'z','NC_FLOAT',dimid);
varid(4) = netcdf.defVar(ncid,'vec_n','NC_FLOAT',dimid3);
varid(5) = netcdf.defVar(ncid,'vec_s1','NC_FLOAT',dimid3);
varid(6) = netcdf.defVar(ncid,'vec_s2','NC_FLOAT',dimid3);
netcdf.endDef(ncid);
netcdf.putVar(ncid,varid(1),x);
netcdf.putVar(ncid,varid(2),y);
netcdf.putVar(ncid,varid(3),z);
netcdf.putVar(ncid,varid(4),vec_n);
netcdf.putVar(ncid,varid(5),vec_m);
netcdf.putVar(ncid,varid(6),vec_l);
netcdf.close(ncid);
