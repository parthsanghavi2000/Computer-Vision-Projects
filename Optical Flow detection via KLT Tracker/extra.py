def estimateFeatureTranslation(feature_x, feature_y, Ix, Iy, img1, img2):
    features_X = feature_x
    features_Y = feature_y

    mesh_x, mesh_y = np.meshgrid(np.arange(WINDOW_SIZE), np.arange(WINDOW_SIZE))

    im1G = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    im2G = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    mesh_x_flat_fix = mesh_x.flatten() + features_X - np.floor(WINDOW_SIZE / 2)
    mesh_y_flat_fix = mesh_y.flatten() + features_Y - np.floor(WINDOW_SIZE / 2)

    coor_fix = np.vstack((mesh_x_flat_fix, mesh_y_flat_fix))

    I1_value = interp2(im1G, coor_fix[[0], :], coor_fix[[1], :])
    Ix_value = interp2(Ix, coor_fix[[0], :], coor_fix[[1], :])
    Iy_value = interp2(Iy, coor_fix[[0], :], coor_fix[[1], :])

    I = np.vstack((Ix_value, Iy_value))
    A = I.dot(I.T)

    for _ in range(15):
        mesh_x_flat = mesh_x.flatten() + features_X - np.floor(WINDOW_SIZE / 2)
        mesh_y_flat = mesh_y.flatten() + features_Y - np.floor(WINDOW_SIZE / 2)
        coor = np.vstack((mesh_x_flat, mesh_y_flat))
        I2_value = interp2(im2G, coor[[0], :], coor[[1], :])
        Ip = (I2_value - I1_value).reshape((-1, 1))
        b = -I.dot(Ip)
        solution = inv(A).dot(b)
        # solution = np.linalg.solve(A, b)
        features_X += solution[0, 0]
        features_Y += solution[1, 0]

    return features_X, features_Y