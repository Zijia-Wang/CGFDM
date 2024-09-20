function [X,Y,Z] = gather_coord_xy(parfile,unit)

%if (nargin < 2)
%  unit = 'm'
%end

par = get_params(parfile);

NX = par.NX;
NY = par.NY;
NZ = par.NZ;
DH = par.DH;
PX = par.PX;
PY = par.PY;
PZ = par.PZ;

ni = NX/PX;
nj = NY/PY;
nk = NZ/PZ;

X = zeros(NY,NX);
Y = zeros(NY,NX);
Z = zeros(NY,NX);

dirnm = par.OUT;

k = PZ-1;
for i = 0:PX-1
for j = 0:PY-1

fnm = [dirnm, '/wave_xy_mpi',...
num2str(i,'%02d'),...
num2str(j,'%02d'),...
num2str(k,'%02d'),'.nc'];

i1 = i * ni + 1; i2 = i1 + ni-1;
j1 = j * nj + 1; j2 = j1 + nj-1;

x = squeeze(ncread(fnm, 'x'));
y = squeeze(ncread(fnm, 'y'));
z = squeeze(ncread(fnm, 'z'));

X(j1:j2,i1:i2) = x;
Y(j1:j2,i1:i2) = y;
Z(j1:j2,i1:i2) = z;

end
end

%if(strcmp(unit, 'km'))
%disp('trans to km')
%X = X * 1e-3;
%Y = Y * 1e-3;
%Z = Z * 1e-3;
%end

end
