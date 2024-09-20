function [X,Y,Z] = gather_coord(parfile)
par = get_params(parfile);

NX = par.NX;
NY = par.NY;
NZ = par.NZ;
DH = par.DH;
PX = par.PX;
PY = par.PY;
PZ = par.PZ;
nfault = par.num_fault;

nj = NY/PY;
nk = NZ/PZ;

X = zeros(NY,NZ,nfault);
Y = zeros(NY,NZ,nfault);
Z = zeros(NY,NZ,nfault);

dirnm = par.OUT;

i = 0;
for j = 0:PY-1
for k = 0:PZ-1

fnm = [dirnm, '/fault_mpi',...
num2str(i,'%02d'),...
num2str(j,'%02d'),...
num2str(k,'%02d'),'.nc'];

j1 = j * nj + 1; j2 = j1 + nj-1;
k1 = k * nk + 1; k2 = k1 + nk-1;

x = squeeze(ncread(fnm, 'x'));
y = squeeze(ncread(fnm, 'y'));
z = squeeze(ncread(fnm, 'z'));

X(j1:j2,k1:k2, :) = x;
Y(j1:j2,k1:k2, :) = y;
Z(j1:j2,k1:k2, :) = z;

end
end

end
