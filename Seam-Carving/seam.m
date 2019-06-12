img1 = imread('tower.jpg');
img = img1; 
iter = 400;
subplot(2,1,1);
imshow(img);
for k = 1:iter
	temp = rgb2gray(img);
	[I,~] = imgradient(temp,'sobel');
	cost = zeros(size(I,1),size(I,2));
	st = 1;
	minc = double(intmax);
	for i = 1:size(I,1)
		for j = 1:size(I,2)
			if i == 1
				cost(i,j) = I(i,j);
			elseif j == 1
				cost(i,j) = I(i,j) + min(cost(i-1,j),cost(i-1,j+1));
			elseif j == size(I,2)
				cost(i,j) = I(i,j) + min(cost(i-1,j),cost(i-1,j-1));
			else
				cost(i,j) = min(cost(i-1,j),cost(i-1,j+1));
				cost(i,j) = min(cost(i,j),cost(i-1,j-1));
				cost(i,j) = cost(i,j) + I(i,j);
			end
			if i == size(I,1)
				if cost(i,j) < minc
					st = j;
					minc = cost(i,j);
				end
			end
		end
	end
	for i = size(I,1):-1:1
		tempst = st;
		if i == 1
			img(i,st:size(I,2)-1,:) = img(i,st+1:size(I,2),:);
		elseif st == 1
			temp = min(cost(i-1,st),cost(i-1,st+1));
			if cost(i-1,st) == temp
				tempst = st;
			else
				tempst = st+1;
			end
		elseif st == size(I,2)
			temp = min(cost(i-1,st),cost(i-1,st-1));
			if cost(i-1,st) == temp
				tempst = st;
			else
				tempst = st-1;
			end
		else
			temp = min(cost(i-1,st),cost(i-1,st-1));
			temp = min(temp,cost(i-1,st+1));
			if cost(i-1,st) == temp
				tempst = st;
			elseif cost(i-1,st-1) == temp
				tempst = st-1;
			else
				tempst = st+1;
			end
		end
		if i ~= 1
			img(i,st:size(I,2)-1,:) = img(i,st+1:size(I,2),:);
		end
		st = tempst;
	end
	tempimg(:,:,:) = img(:,1:end-1,:); 
	img = tempimg;
	clear tempimg;
end
subplot(2,1,2);
imshow(img);