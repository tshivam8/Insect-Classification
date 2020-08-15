%https://in.mathworks.com/matlabcentral/answers/116793-how-to-classify-shapes-of-this-image-as-square-rectangle-triangle-and-circle
%https://in.mathworks.com/matlabcentral/answers/uploaded_files/8433/shape_recognition.m
% Demo to find certain shapes in an image based on their shape.
warning off;
clc;    % Clear the command window.
clear all;
close all;  % Close all figures (except those of imtool.)
imtool close all;  % Close all imtool figures.
clear;  % Erase all existing variables.
workspace;  % Make sure the workspace panel is showing.
fontSize = 20;
clc;
clear all;
warning off;
% the folder in which ur images exists
srcFiles = dir('D:\JOURNALS\11_ML DL_Information processing in Agriculture\ML paper 2020 new\Xie-24 classes\Xie24 - resized - Augmented\TRAIN\24_Nephotettix bipunctatus\*.jpg');
%  srcFiles = dir('D:\JOURNALS\11_ML DL_Information processing in Agriculture\ML paper 2020 new\Wang-9 classes\Wang 9 classes_resized 227x227\Wang 9 classes_Augmented\Orthoptera\*.jpg');
for i = 1 : length(srcFiles)
    filename = strcat('D:\JOURNALS\11_ML DL_Information processing in Agriculture\ML paper 2020 new\Xie-24 classes\Xie24 - resized - Augmented\TRAIN\24_Nephotettix bipunctatus\',srcFiles(i).name);

    
%     filename = strcat('D:\JOURNALS\11_ML DL_Information processing in Agriculture\ML paper 2020 new\Wang-9 classes\Wang 9 classes_resized 227x227\Wang 9 classes_Augmented\Orthoptera\',srcFiles(i).name);
    I = imread(filename);
    subplot(20,20,i);
    imshow(I);
   end
figure
  for j = 1 : length(srcFiles)
            filename = strcat('D:\JOURNALS\11_ML DL_Information processing in Agriculture\ML paper 2020 new\Xie-24 classes\Xie24 - resized - Augmented\TRAIN\24_Nephotettix bipunctatus\',srcFiles(j).name);

%       filename = strcat('D:\JOURNALS\11_ML DL_Information processing in Agriculture\ML paper 2020 new\Wang-9 classes\Wang 9 classes_resized 227x227\Wang 9 classes_Augmented\Orthoptera\',srcFiles(j).name);
    I = imread(filename);  
 hedge = vision.EdgeDetector;
 hcsc = vision.ColorSpaceConverter('Conversion', 'RGB to intensity');
 hidtypeconv = vision.ImageDataTypeConverter('OutputDataType','single');
 img = step(hcsc, I);
 img1 = step(hidtypeconv, img);
%J = wiener2(img1,[9,9]);
 edges1 = step(hedge, img1);
 %edges=bwareaopen(edges1,5,4);
 %imshow(edges);
 %figure,imshow(img1);
%  figure,imshow(J);
        se90=strel('line',4,90);
        se0=strel('line',4,90);
        I1=imdilate(edges1,[se90,se0]);
        se = strel('disk',10);
        I2=imclose(I1,se);
        %subplot(223);
        %imshow(I2);
        %title('close fn o/p');
        %subplot(5,5,z);
%         subplot(8,8,j);
       filledImage=imfill(I2,'holes');
 imshow(filledImage);  
  
% figure

[labeledImage numberOfObjects] = bwlabel(filledImage);
%blobMeasurements = regionprops(labeledImage,...
	%'Perimeter', 'Area', 'FilledArea', 'Solidity', 'Centroid'); 
blobMeasurements = regionprops(labeledImage,...
	 'Centroid', 'Area', 'Perimeter','Eccentricity','FilledArea', 'Extent','Solidity' ,'MajorAxisLength','MinorAxisLength');
% Get the outermost boundaries of the objects, just for fun.
%filledImage = imfill(binaryImage, 'holes');
boundaries = bwboundaries(filledImage);

% Collect some of the measurements into individual arrays.
centroids= [blobMeasurements.Centroid];
areas = [blobMeasurements.Area];
perimeters = [blobMeasurements.Perimeter];
eccentricity = [blobMeasurements.Eccentricity];
filledAreas = [blobMeasurements.FilledArea];
solidities = [blobMeasurements.Solidity];
majoraxislength = [blobMeasurements.MajorAxisLength];
minoraxislength = [blobMeasurements.MinorAxisLength];
circularities = (perimeters .^2)./ (4 * pi * filledAreas);
formfactor=(4*pi*filledAreas)./(perimeters.^2);
roundness=(4*areas)./(pi*(majoraxislength.^2));
aspectratio=majoraxislength/minoraxislength;
compactness=(sqrt((4/pi)*filledAreas))./majoraxislength;
extent = [blobMeasurements.Extent];

% Print to command window.


featureMatrix=zeros(numberOfObjects,9);
% featureMatrix(1,1)= areas(1);
% featureMatrix(1,2)= perimeters(1);
% featureMatrix(1,3)= eccentricity(1);
% featureMatrix(1,4)= circularities(1);
% featureMatrix(1,5)= solidities(1);
col_header={'','Area','Perimeter','Major axis length','Minor axis length','Eccentricity','Circularity','Solidity','Form factor','Compactness'};     %Row cell array (for column labels)
row_header={'Insect'};

for i = 1 : numberOfObjects 
    featureMatrix(i,1) = areas(i);
    featureMatrix(i,2) = perimeters(i);
    featureMatrix(i,3) = majoraxislength(i);
     featureMatrix(i,4) = minoraxislength(i);
      featureMatrix(i,5) = eccentricity(i);
    featureMatrix(i,6) = circularities(i);
    featureMatrix(i,7) = solidities(i);
    featureMatrix(i,8) = formfactor(i);
    featureMatrix(i,9) = compactness(i);
        %header={'Area','Perimeter','Eccentricity','Circularity','Solidity'};
         %Column cell array (for row labels)
     str1='B';
     str2=int2str(j+1);
     str=strcat(str1,str2);
     str11='A';
     
     strr=strcat(str11,str2);
     
% xlswrite('D:\MATLAB PROGRAMS\Shapes\b.xlsx',data,'Sheet1','B2');     %Write data
 xlswrite('D:\MATLAB PROGRAMS\Insect_Image Processing_Matlab\InsectShapeFeaturesResult2.xlsx',[areas(i),perimeters(i),majoraxislength(i),minoraxislength(i),eccentricity(i),circularities(i),solidities(i),formfactor(i),compactness(i)]); 
 xlswrite('D:\MATLAB PROGRAMS\Insect_Image Processing_Matlab\InsectShapeFeaturesResult2.xlsx',col_header,'Sheet1','A1');,     %Write column header
 xlswrite('D:\MATLAB PROGRAMS\Insect_Image Processing_Matlab\InsectShapeFeaturesResult2.xlsx',row_header,'Sheet1',strr);
 xlswrite('D:\MATLAB PROGRAMS\Insect_Image Processing_Matlab\InsectShapeFeaturesResult2.xlsx',featureMatrix,'Sheet1',str);
    %featureMatrix(i,6) = majoraxislength(i);
    %featureMatrix(i,7) = minoraxislength(i)

end
  end

 