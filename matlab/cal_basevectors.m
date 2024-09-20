function [n,m,l] = cal_basevectors(metric);

[ny, nz] = size(metric.xix);

n = zeros(3, ny, nz);
m = zeros(3, ny, nz);
l = zeros(3, ny, nz);

n(1,:,:) = metric.xix;
n(2,:,:) = metric.xiy;
n(3,:,:) = metric.xiz;

m(1,:,:) = metric.x_et;
m(2,:,:) = metric.y_et;
m(3,:,:) = metric.z_et;

n_norm=sqrt(n(1,:,:).^2+n(2,:,:).^2+n(3,:,:).^2);
m_norm=sqrt(m(1,:,:).^2+m(2,:,:).^2+m(3,:,:).^2);
n = n./repmat(n_norm, 3, 1, 1);
m = m./repmat(m_norm, 3, 1, 1);

l(1,:,:) = n(2,:,:).*m(3,:,:)-n(3,:,:).*m(2,:,:);
l(2,:,:) = n(3,:,:).*m(1,:,:)-n(1,:,:).*m(3,:,:);
l(3,:,:) = n(1,:,:).*m(2,:,:)-n(2,:,:).*m(1,:,:);

end
