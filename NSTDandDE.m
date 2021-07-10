function [f1, f2] = NSTDandDE(P,I,Crop)

P2=imcrop(P,Crop);
Prom=(sum(sum(P2.*I)))/(sum(sum(P2.^2)));
f1=(sum(sum((I-P2.*Prom).^2))/(sum(sum(I.^2))))^0.5;
f2=sum(sum(P2))/sum(sum(P));
