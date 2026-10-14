function imageData = rowToImage(pixelRow)
%ROWTOIMAGE Convert a 1-by-784 MNIST row into a 28-by-28 image.
% Preserves pixel values and the orientation used by the saved MNIST data.
% Example: imagesc(rowToImage(xFit(1,:))); colormap gray; axis image;

    validateattributes(pixelRow, {'numeric', 'logical'}, {'size', [1, 784]}, ...
        mfilename, 'pixelRow');
    imageData = reshape(pixelRow, 28, 28);
end
