clc
clear
addpath('../../matlab/');
par = get_params('params.json');

ny = par.NY
nz = par.NZ
dh = par.DH
OUT = par.OUT

x = zeros(ny, nz);
y = zeros(ny, nz);
z = zeros(ny, nz);

for j = 1:ny
    for k = 1:nz
        x(j,k) = 0;
        y(j,k) = (j-1-ny/2)*dh;
        z(j,k) = (k-nz)*dh;
    end
end

if 0
for j = 1:ny
    for k = 1:nz
        r1 = sqrt((y(j,k)+10.5e3).^2 + (z(j,k)+7.5e3).^2);
        r2 = sqrt((y(j,k)-10.5e3).^2 + (z(j,k)+7.5e3).^2);
        fxy = 0;
        if(r1 <3e3)
            fxy = 300 * (1+cos(pi*r1/3e3));
        end
        
        if(r2 <3e3)
            fxy = 300 * (1+cos(pi*r2/3e3));
        end
        
        x(j,k)=x(j,k)+fxy;
    end
end
end

if 0
figure
pcolor(y, z, x);
shading flat
colorbar
axis image
end

disp('calculating metric and base vectors...')
metric = cal_metric(x,y,z, dh);
[vec_n, vec_m, vec_l] = cal_basevectors(metric);


if 0
u = squeeze(vec_n(1,:,:))*1e3;
v = squeeze(vec_n(2,:,:))*1e3;
w = squeeze(vec_n(3,:,:))*1e3;
s=10;
j=1:s:ny;
k=1:s:nz;
figure
quiver3(x(j,k),y(j,k),z(j,k),u(j,k),v(j,k),w(j,k));
end

disp('write output...')
fnm_out = par.Fault_geometry
ncid = netcdf.create(fnm_out, 'CLOBBER');
dimid(1) = netcdf.defDim(ncid,'ny',ny);
dimid(2) = netcdf.defDim(ncid,'nz',nz);
dimid3(1) = netcdf.defDim(ncid, 'dim', 3);
dimid3(2) = dimid(1);
dimid3(3) = dimid(2);
varid(1) = netcdf.defVar(ncid,'x','NC_FLOAT',dimid);
varid(2) = netcdf.defVar(ncid,'y','NC_FLOAT',dimid);
varid(3) = netcdf.defVar(ncid,'z','NC_FLOAT',dimid);
varid(4) = netcdf.defVar(ncid,'vec_n','NC_FLOAT',dimid3);
varid(5) = netcdf.defVar(ncid,'vec_m','NC_FLOAT',dimid3);
varid(6) = netcdf.defVar(ncid,'vec_l','NC_FLOAT',dimid3);
netcdf.endDef(ncid);
netcdf.putVar(ncid,varid(1),x);
netcdf.putVar(ncid,varid(2),y);
netcdf.putVar(ncid,varid(3),z);
netcdf.putVar(ncid,varid(4),vec_n);
netcdf.putVar(ncid,varid(5),vec_m);
netcdf.putVar(ncid,varid(6),vec_l);
netcdf.close(ncid);
