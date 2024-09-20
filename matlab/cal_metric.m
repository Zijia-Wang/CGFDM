function metric = cal_metric(x, y, z, dh);

[ny, nz] = size(x);
dh2 = 2*dh;

%X = repmat(x, [2, ny, nz]);
%Y = repmat(y, [2, ny, nz]);
%Z = repmat(z, [2, ny, nz]);
%X(1,:,:) = x-dh;
%X(2,:,:) = x+dh;

x_xi = zeros(ny, nz);
y_xi = zeros(ny, nz);
z_xi = zeros(ny, nz);
x_et = zeros(ny, nz);
y_et = zeros(ny, nz);
z_et = zeros(ny, nz);
x_zt = zeros(ny, nz);
y_zt = zeros(ny, nz);
z_zt = zeros(ny, nz);
jac = zeros(ny, nz);
xix = zeros(ny,nz);
xiy = zeros(ny,nz);
xiz = zeros(ny,nz);
etx = zeros(ny,nz);
ety = zeros(ny,nz);
etz = zeros(ny,nz);
ztx = zeros(ny,nz);
zty = zeros(ny,nz);
ztz = zeros(ny,nz);

x_xi(:,:) = 1;%(x(2,:,:)-x(1,:,:))/dh2; % 1
y_xi(:,:) = 0;%(y(2,:,:)-y(1,:,:))/dh2; % 0
z_xi(:,:) = 0;%(z(2,:,:)-z(1,:,:))/dh2; % 0

j = 2:ny-1;
k = 2:nz-1;

x_et(j,k) = (x(j+1,k)-x(j-1,k))/dh2;
y_et(j,k) = (y(j+1,k)-y(j-1,k))/dh2;
z_et(j,k) = (z(j+1,k)-z(j-1,k))/dh2;

x_zt(j,k) = (x(j,k+1)-x(j,k-1))/dh2;
y_zt(j,k) = (y(j,k+1)-y(j,k-1))/dh2;
z_zt(j,k) = (z(j,k+1)-z(j,k-1))/dh2;

x_et = extend_symm(x_et);
y_et = extend_symm(y_et);
z_et = extend_symm(z_et);

x_zt = extend_symm(x_zt);
y_zt = extend_symm(y_zt);
z_zt = extend_symm(z_zt);

jac = ...
x_xi.*y_et.*z_zt+...
x_zt.*y_xi.*z_et+...
x_et.*y_zt.*z_xi-...
x_xi.*y_zt.*z_et-...
x_et.*y_xi.*z_zt-...
x_zt.*y_et.*z_xi;

for j = 1:ny-0
for k = 1:nz-0
M = [
x_xi(j,k) x_et(j,k) x_zt(j,k);
y_xi(j,k) y_et(j,k) y_zt(j,k);
z_xi(j,k) z_et(j,k) z_zt(j,k);
];
N = inv(M);
xix(j,k) = N(1,1);
xiy(j,k) = N(1,2);
xiz(j,k) = N(1,3);

etx(j,k) = N(2,1);
ety(j,k) = N(2,2);
etz(j,k) = N(2,3);

ztx(j,k) = N(3,1);
zty(j,k) = N(3,2);
ztz(j,k) = N(3,3);
end
end
% accelerate by bsxfun
%I = eye(3);
%M = zeros(3,3,ny,nz);
%M(1,1,:,:) = x_xi;
%M(2,1,:,:) = y_xi;
%M(3,1,:,:) = z_xi;
%M(1,2,:,:) = x_et;
%M(2,2,:,:) = y_et;
%M(3,2,:,:) = z_et;
%M(1,3,:,:) = x_zt;
%M(2,3,:,:) = y_zt;
%M(3,3,:,:) = z_zt;
%N = bsxfun(@mrdivide, I, M);
%xix = squeeze(N(1,1,:,:));
%xiy = squeeze(N(1,2,:,:));
%xiz = squeeze(N(1,3,:,:));
%etx = squeeze(N(2,1,:,:));
%ety = squeeze(N(2,2,:,:));
%etz = squeeze(N(2,3,:,:));
%ztx = squeeze(N(3,1,:,:));
%zty = squeeze(N(3,2,:,:));
%ztz = squeeze(N(3,3,:,:));

metric.x_xi = x_xi;
metric.y_xi = y_xi;
metric.z_xi = z_xi;
metric.x_et = x_et;
metric.y_et = y_et;
metric.z_et = z_et;
metric.x_zt = x_zt;
metric.y_zt = y_zt;
metric.z_zt = z_zt;
metric.jac = jac;
metric.xix = xix;
metric.xiy = xiy;
metric.xiz = xiz;
metric.etx = etx;
metric.ety = ety;
metric.etz = etz;
metric.ztx = ztx;
metric.zty = zty;
metric.ztz = ztz;

end

function a = extend_symm(a)
[m, n] = size(a);
a(1,:) = a(2,:);
a(m,:) = a(m-1,:);
a(:,1) = a(:,2);
a(:,n) = a(:,n-1);
end
