clear
close all;

Name='1_256';    %Input file name

alpha=0.9;  %0 - only DE, 1 - only NSTD

% Hologram size
SizeX=1024;
SizeY=1024;

dH=10.8*10^(-6);    %Hologram pixel size
l=633*10^(-9);      %Wavelength

MaxNiter = 50;      %Number ofiterations to perform
    
Sparsing=0;         %0 or 1 to toggle on/off sparsing of input image

FocusLense=0.3;     %Distanse between point sourse and hologram;



GS=strcat(Name,'.tif');
RandStream.setGlobalStream (RandStream('mt19937ar','seed',15));
N1=SizeY;
N2=SizeX;
N1_Rec=N1-1;
N2_Rec=N2-1;
[a0,b0] = meshgrid(-N2_Rec/2 : N2_Rec/2, -N1_Rec/2 : N1_Rec/2);

J = imread (GS);
J=double(J);
J=imresize(J, [size(J,1) round(size(J,2)*N2/N1)]);
if Sparsing>0
    New=zeros(size(J)*(Sparsing+1));
    New(1:Sparsing+1:size(J,1)*(Sparsing+1),1:Sparsing+1:size(J,2)*(Sparsing+1))=J(1:1:size(J,1),1:1:size(J,2));
    J=New;
end

XY(1)=floor((SizeY-size(J,1))/2)+1;
XY(2)=floor((SizeX-size(J,2))/2)+1;

Crop=[XY(2) XY(1) size(J,2)-1 size(J,1)-1];
SourceSize = size(J);

J12=J.^0.5;
J1 = zeros(SizeY,SizeX);
J1(XY(1):1:SourceSize(1)+XY(1)-1,XY(2):1:SourceSize(2)+XY(2)-1)=J(1:1:SourceSize(1),1:1:SourceSize(2));

imwrite(uint8(J1),strcat(Name,num2str(SizeX),'x',num2str(SizeY),'.tif'), 'tif')
 
new =(J1).^0.5; %Интенсивность пропорциональна квадрату амплитуды 

s=size(J1);
arg2 = rand(size(J1));

LightPhase=angle(double(exp(((-pi*1i/(FocusLense*l)).*(dH*dH).*(a0.*a0 + b0.*b0))))); 
    
ii = 0;
DoNextIter = true;

absffund = ones(s(1),s(2));
LightAmplitude=absffund;
for i=1:N1
    for j=1:N2
        LightAmplitude(i,j)=(FocusLense^2/(FocusLense^2+(dH^2)*((i-N1/2)^2+(j-N2/2)^2))).^0.5;
    end
end

% Light distribution in holgram plane
newffund = LightAmplitude.*arg2.*exp(1i.*LightPhase);

% Reconstruction plane:
newfund =ifftshift(ifft2(fftshift(newffund)));
    
while (ii < MaxNiter) && (DoNextIter) 
    ii = ii+1;
    arg1=angle(newfund);
    Jnew = abs(newfund);
    Jnew2=imcrop(Jnew, [XY(2) XY(1) SourceSize(2)-1 SourceSize(1)-1]);
    Jnew(XY(1):1:SourceSize(1)+XY(1)-1,XY(2):1:SourceSize(2)+XY(2)-1)=Jnew2(1:1:SourceSize(1),1:1:SourceSize(2))*0+J12(1:1:SourceSize(1),1:1:SourceSize(2))*mean2(Jnew2)/mean2(J12)*1;
    newfuni = Jnew.*exp(1i.*arg1);
    ffund = ifftshift(fft2(fftshift(newfuni)));
    ffund=ffund.*exp(-1i.*LightPhase)./LightAmplitude;
    Holo=abs(ffund);
    Holo0=Holo;
    Mean=mean2(Holo);
    Holo = uint8((Holo/max(Holo(:)))*255);
    level = graythresh(Holo);
    Holo = im2bw(Holo,level);        
    Holo=double(Holo);
    Holo=Holo/mean2(Holo)*Mean;

    %     Light distribution in holgram plane
    newffund = LightAmplitude.*Holo0.*exp(1i.*LightPhase);
    
    %     Reconstruction plane:
    newfund =ifftshift(ifft2(fftshift(newffund)));
    
    Jnew = LightAmplitude.*Holo.*exp(1i.*LightPhase);
    Jnew =ifftshift(ifft2(fftshift(Jnew)));  
    Jnew = Jnew.*conj(Jnew);
    G = Jnew./max(Jnew(:)).*255; %Нормировка необязательна 

    % Calculating NSTD and diffraction efficiency
    [NSTD,DE]=NSTDandDE(G,J,Crop);
    DE=DE*mean2(Holo)/max(Holo(:));
    TF=alpha*NSTD+(1-alpha)*(1-DE);
    if ii==1
        MinTF=TF;
        BestNSTD=NSTD;
        BestDE=DE;
        BestK=Holo;
        BestKR=Jnew;
        BestIter=ii;
    else
        if MinTF>TF
            MinTF=TF;
            BestK=Holo;
            BestKR=Jnew;
            BestIter=ii;
            BestNSTD=NSTD;
            BestDE=DE;
        end
    end
    NumIter = ii;

    figure (1)
    plot(ii,TF,'rs','MarkerFaceColor','r','MarkerSize',3)   
    hold on
    plot(ii,NSTD,'bd','MarkerFaceColor','b','MarkerSize',3)
    plot(ii,DE,'g^','MarkerFaceColor','g','MarkerSize',3)
    legend('Target function', 'NSTD','Diffraction efficiency')
    drawnow  
    grid on
    xlabel('Iteration number')
    ylabel('Target Function')
    disp(strcat('Iteration_',num2str(ii),'. NSTD=',num2str(NSTD),'_DE=',num2str(DE),'_TF=',num2str(TF),'_Best TF=',num2str(MinTF),'_Best itearation=',num2str(BestIter)))
end

% Saving to file

Named2=strcat('GS_D_F',num2str(FocusLense),'_',Name,num2str(SizeX),'x',num2str(SizeY),'K_GS_F_D_','NSTD',num2str(BestNSTD),'_DE',num2str(BestDE),'BI',num2str(BestIter),'Sp',num2str(Sparsing));
Named3=strcat(Named2,'PSF');

newffund2 = (BestK/max(BestK(:)))*255;
     
imwrite(uint8(newffund2),strcat(Named2,'.bmp'),'bmp');  %Hologram
Gu=BestKR/max(BestKR(:))*65535;
imwrite(uint16(Gu),strcat(Named3,'.tif'),'tif');        %Reconstruction plane