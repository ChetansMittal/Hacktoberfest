{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Untitled1.ipynb",
      "provenance": [],
      "authorship_tag": "ABX9TyOn5iVvw8oLZFLIg7JHa7di",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/ChetansMittal/Handwritten-Digit-Recognition/blob/main/HandWriiten_Digit_Recognition.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "SmVMCwbLGkyL"
      },
      "source": [
        "import numpy as np\n",
        "from sklearn.datasets import load_digits\n",
        "\n"
      ],
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "_DYP_FmoNx69"
      },
      "source": [
        "dataset=load_digits()"
      ],
      "execution_count": 3,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "IAJ6pHHtN5hf",
        "outputId": "2083df46-9174-4407-a406-342238c43b07"
      },
      "source": [
        "print(dataset.data)\n",
        "print(dataset.target)\n",
        "\n",
        "print(dataset.data.shape)\n",
        "print(dataset.images.shape)\n",
        "\n",
        "dataimageLength =len(dataset.images)\n",
        "print(dataimageLength)"
      ],
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[ 0.  0.  5. ...  0.  0.  0.]\n",
            " [ 0.  0.  0. ... 10.  0.  0.]\n",
            " [ 0.  0.  0. ... 16.  9.  0.]\n",
            " ...\n",
            " [ 0.  0.  1. ...  6.  0.  0.]\n",
            " [ 0.  0.  2. ... 12.  0.  0.]\n",
            " [ 0.  0. 10. ... 12.  1.  0.]]\n",
            "[0 1 2 ... 8 9 8]\n",
            "(1797, 64)\n",
            "(1797, 8, 8)\n",
            "1797\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 435
        },
        "id": "Ipkv0w5GPNKC",
        "outputId": "a0cfeb11-820b-40e2-b6f4-349d293da5f7"
      },
      "source": [
        "n=9\n",
        " \n",
        "\n",
        "import matplotlib.pyplot as plt\n",
        "plt.gray()\n",
        "plt.matshow(dataset.images[n])\n",
        "plt.show()\n",
        "\n",
        "dataset.images[n]"
      ],
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 0 Axes>"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 288x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "needs_background": "light"
          }
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[ 0.,  0., 11., 12.,  0.,  0.,  0.,  0.],\n",
              "       [ 0.,  2., 16., 16., 16., 13.,  0.,  0.],\n",
              "       [ 0.,  3., 16., 12., 10., 14.,  0.,  0.],\n",
              "       [ 0.,  1., 16.,  1., 12., 15.,  0.,  0.],\n",
              "       [ 0.,  0., 13., 16.,  9., 15.,  2.,  0.],\n",
              "       [ 0.,  0.,  0.,  3.,  0.,  9., 11.,  0.],\n",
              "       [ 0.,  0.,  0.,  0.,  9., 15.,  4.,  0.],\n",
              "       [ 0.,  0.,  9., 12., 13.,  3.,  0.,  0.]])"
            ]
          },
          "metadata": {},
          "execution_count": 7
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "bAdDlyf2RgsM",
        "outputId": "8b2ec946-7cb2-4959-bd2d-eaa32026dd56"
      },
      "source": [
        "X=dataset.images.reshape((dataimageLength,-1))\n",
        "X"
      ],
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[ 0.,  0.,  5., ...,  0.,  0.,  0.],\n",
              "       [ 0.,  0.,  0., ..., 10.,  0.,  0.],\n",
              "       [ 0.,  0.,  0., ..., 16.,  9.,  0.],\n",
              "       ...,\n",
              "       [ 0.,  0.,  1., ...,  6.,  0.,  0.],\n",
              "       [ 0.,  0.,  2., ..., 12.,  0.,  0.],\n",
              "       [ 0.,  0., 10., ..., 12.,  1.,  0.]])"
            ]
          },
          "metadata": {},
          "execution_count": 8
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "w8jvIZhXR8nc",
        "outputId": "5b18700a-e973-451d-fc11-4a6a3a464cdd"
      },
      "source": [
        "Y=dataset.target\n",
        "Y"
      ],
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([0, 1, 2, ..., 8, 9, 8])"
            ]
          },
          "metadata": {},
          "execution_count": 9
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "yMUI1rLOSCAI",
        "outputId": "023116b9-9740-4999-c701-5b3f2a9e509f"
      },
      "source": [
        "from sklearn.model_selection import  train_test_split\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size= 0.25, random_state=0)\n",
        "print(X_train.shape)\n",
        "print(X_test.shape )"
      ],
      "execution_count": 12,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "(1347, 64)\n",
            "(450, 64)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "udt-dBkVTIF8",
        "outputId": "2c23c153-8ee2-4eb3-cc6f-54180aab5745"
      },
      "source": [
        "from sklearn import svm\n",
        "model = svm.SVC(kernel='linear')\n",
        "model.fit(X_train, y_train)"
      ],
      "execution_count": 11,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,\n",
              "    decision_function_shape='ovr', degree=3, gamma='scale', kernel='linear',\n",
              "    max_iter=-1, probability=False, random_state=None, shrinking=True,\n",
              "    tol=0.001, verbose=False)"
            ]
          },
          "metadata": {},
          "execution_count": 11
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 317
        },
        "id": "3Ki7WTXJWCWQ",
        "outputId": "cba08522-1f21-40a9-f8f2-46ea0dbc593c"
      },
      "source": [
        "n=585\n",
        "result= model.predict(dataset.images[n].reshape(1,-1))\n",
        "plt.imshow(dataset.images[n], cmap=plt.cm.gray_r, interpolation='nearest')\n",
        "print(result)\n",
        "print(\"\\n\")\n",
        "plt.axis(\"off\") \n",
        "plt.title('%i' %result)\n",
        "plt.show()"
      ],
      "execution_count": 17,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[9]\n",
            "\n",
            "\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAOcAAAD3CAYAAADmIkO7AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAE4UlEQVR4nO3dsYqdWxmA4W9JBDkgBDu7qQThcJg7cIJoG7wAYUCwTQothdxBBk51ugl4A4KVhaQSBSHTChZJJWgTy1P9FsdyOIEE1n6z8zzl3sX3sWde1j/DMGsdxzFAz3dOvQBwP3FClDghSpwQJU6IEidEiROixHkm1lo/Xmv9ea3137XWP9davzj1TnwYcZ6BtdaDmfnDzPxxZn4wM7+emd+vtX500sX4IMtfCH381lqfz8xfZ+b7x/+/oGutP83M347j+N1Jl+O9OTnP15qZz0+9BO9PnOfhHzPz75n57Vrru2utn8/MT2bms9OuxYfwWHsm1lpfzMyX881p+feZ+c/MfH0cx69OuhjvTZxnaq31l5l5cRzHV6fehffjsfZMrLW+WGt9b6312VrrNzPzw5m5PfFafABxno9fzsy/5pufPX86Mz87juPr067Eh/BYC1FOTogSJ0SJE6LECVEP3vH+Wf626O3bt1vn3dzcbJt1e3u7bdbOz/HFixfbZs3MPH78eOe4dd+LTk6IEidEiROixAlR4oQocUKUOCFKnBAlTogSJ0SJE6LECVHihChxQpQ4IUqcECVOiBInRIkTosQJUeKEKHFClDghSpwQJU6IEidEves6hm12/mv/y8vLbbNmZq6vr7fNuru72zZr5+f46tWrbbNmtl/HcC8nJ0SJE6LECVHihChxQpQ4IUqcECVOiBInRIkTosQJUeKEKHFClDghSpwQJU6IEidEiROixAlR4oQocUKUOCFKnBAlTogSJ0SJE6Iy1zHc3t5um3VxcbFt1szMs2fPts26ubnZNuvNmzfbZn2KnJwQJU6IEidEiROixAlR4oQocUKUOCFKnBAlTogSJ0SJE6LECVHihChxQpQ4IUqcECVOiBInRIkTosQJUeKEKHFClDghSpwQJU6IEidEZe5Kuby83DZr530iMzNXV1db5+3y5MmTU69w1pycECVOiBInRIkTosQJUeKEKHFClDghSpwQJU6IEidEiROixAlR4oQocUKUOCFKnBAlTogSJ0SJE6LECVHihChxQpQ4IUqcECVOiMpcx7DzyoK7u7tts2ZmXr9+vW3Wzmstnj59um3Wp8jJCVHihChxQpQ4IUqcECVOiBInRIkTosQJUeKEKHFClDghSpwQJU6IEidEiROixAlR4oQocUKUOCFKnBAlTogSJ0SJE6LECVHihKjMdQw7PXz4cOu8nVck7PTy5cttsz7Fqx+cnBAlTogSJ0SJE6LECVHihChxQpQ4IUqcECVOiBInRIkTosQJUeKEKHFClDghSpwQJU6IEidEiROixAlR4oQocUKUOCFKnBAlTohax3F82/vf+ubH6urqauu8i4uLbbOur6+3zXr06NG2We/4Pv3YrftedHJClDghSpwQJU6IEidEiROixAlR4oQocUKUOCFKnBAlTogSJ0SJE6LECVHihChxQpQ4IUqcECVOiBInRIkTosQJUeKEKHFClDghSpwQ9eDUC5zC7e3t2c7beX/J8+fPt836FDk5IUqcECVOiBInRIkTosQJUeKEKHFClDghSpwQJU6IEidEiROixAlR4oQocUKUOCFKnBAlTogSJ0SJE6LECVHihChxQpQ4IUqcELWO4zj1DsA9nJwQJU6IEidEiROixAlR4oSo/wFzAns0daOqnwAAAABJRU5ErkJggg==\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "SiRjDeA4Xpfy",
        "outputId": "4e9f2317-aed0-41b7-a0cd-347364ff58e9"
      },
      "source": [
        "y_pred=model.predict(X_test)\n",
        "print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)) ,1))\n"
      ],
      "execution_count": 18,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[2 2]\n",
            " [8 8]\n",
            " [2 2]\n",
            " [6 6]\n",
            " [6 6]\n",
            " [7 7]\n",
            " [1 1]\n",
            " [9 9]\n",
            " [8 8]\n",
            " [5 5]\n",
            " [2 2]\n",
            " [8 8]\n",
            " [6 6]\n",
            " [6 6]\n",
            " [6 6]\n",
            " [6 6]\n",
            " [1 1]\n",
            " [0 0]\n",
            " [5 5]\n",
            " [8 8]\n",
            " [8 8]\n",
            " [7 7]\n",
            " [8 8]\n",
            " [4 4]\n",
            " [7 7]\n",
            " [5 5]\n",
            " [4 4]\n",
            " [9 9]\n",
            " [2 2]\n",
            " [9 9]\n",
            " [4 4]\n",
            " [7 7]\n",
            " [6 6]\n",
            " [8 8]\n",
            " [9 9]\n",
            " [4 4]\n",
            " [3 3]\n",
            " [1 1]\n",
            " [0 0]\n",
            " [1 1]\n",
            " [8 8]\n",
            " [6 6]\n",
            " [7 7]\n",
            " [7 7]\n",
            " [1 1]\n",
            " [0 0]\n",
            " [7 7]\n",
            " [6 6]\n",
            " [2 2]\n",
            " [1 1]\n",
            " [9 9]\n",
            " [6 6]\n",
            " [7 7]\n",
            " [9 9]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [5 5]\n",
            " [1 1]\n",
            " [6 6]\n",
            " [3 3]\n",
            " [0 0]\n",
            " [2 2]\n",
            " [3 3]\n",
            " [4 4]\n",
            " [1 1]\n",
            " [9 9]\n",
            " [2 2]\n",
            " [6 6]\n",
            " [9 9]\n",
            " [1 1]\n",
            " [8 8]\n",
            " [3 3]\n",
            " [5 5]\n",
            " [1 1]\n",
            " [2 2]\n",
            " [8 8]\n",
            " [2 2]\n",
            " [2 2]\n",
            " [9 9]\n",
            " [7 7]\n",
            " [2 2]\n",
            " [3 3]\n",
            " [6 6]\n",
            " [0 0]\n",
            " [5 5]\n",
            " [3 3]\n",
            " [7 7]\n",
            " [5 5]\n",
            " [1 1]\n",
            " [2 2]\n",
            " [9 9]\n",
            " [9 9]\n",
            " [3 3]\n",
            " [1 1]\n",
            " [7 7]\n",
            " [7 7]\n",
            " [4 4]\n",
            " [8 8]\n",
            " [5 5]\n",
            " [8 8]\n",
            " [5 5]\n",
            " [5 5]\n",
            " [2 2]\n",
            " [5 5]\n",
            " [9 9]\n",
            " [0 0]\n",
            " [7 7]\n",
            " [1 1]\n",
            " [4 4]\n",
            " [4 7]\n",
            " [3 3]\n",
            " [4 4]\n",
            " [8 8]\n",
            " [9 9]\n",
            " [7 7]\n",
            " [9 9]\n",
            " [8 8]\n",
            " [2 2]\n",
            " [1 6]\n",
            " [5 5]\n",
            " [2 2]\n",
            " [5 5]\n",
            " [8 8]\n",
            " [4 4]\n",
            " [1 8]\n",
            " [7 7]\n",
            " [0 0]\n",
            " [6 6]\n",
            " [1 1]\n",
            " [5 5]\n",
            " [5 9]\n",
            " [9 9]\n",
            " [9 9]\n",
            " [5 5]\n",
            " [9 9]\n",
            " [9 9]\n",
            " [5 5]\n",
            " [7 7]\n",
            " [5 5]\n",
            " [6 6]\n",
            " [2 2]\n",
            " [8 8]\n",
            " [6 6]\n",
            " [9 9]\n",
            " [6 6]\n",
            " [1 1]\n",
            " [5 5]\n",
            " [1 1]\n",
            " [5 5]\n",
            " [9 9]\n",
            " [9 9]\n",
            " [1 1]\n",
            " [5 5]\n",
            " [3 3]\n",
            " [6 6]\n",
            " [1 1]\n",
            " [8 8]\n",
            " [9 9]\n",
            " [8 8]\n",
            " [7 7]\n",
            " [6 6]\n",
            " [7 7]\n",
            " [6 6]\n",
            " [5 5]\n",
            " [6 6]\n",
            " [0 0]\n",
            " [8 8]\n",
            " [8 8]\n",
            " [9 9]\n",
            " [8 8]\n",
            " [6 6]\n",
            " [1 1]\n",
            " [0 0]\n",
            " [4 4]\n",
            " [1 1]\n",
            " [6 6]\n",
            " [3 3]\n",
            " [8 8]\n",
            " [6 6]\n",
            " [7 7]\n",
            " [4 4]\n",
            " [9 5]\n",
            " [6 6]\n",
            " [3 3]\n",
            " [0 0]\n",
            " [3 3]\n",
            " [3 3]\n",
            " [3 3]\n",
            " [0 0]\n",
            " [7 7]\n",
            " [7 7]\n",
            " [5 5]\n",
            " [7 7]\n",
            " [8 8]\n",
            " [0 0]\n",
            " [7 7]\n",
            " [1 8]\n",
            " [9 9]\n",
            " [6 6]\n",
            " [4 4]\n",
            " [5 5]\n",
            " [0 0]\n",
            " [1 1]\n",
            " [4 4]\n",
            " [6 6]\n",
            " [4 4]\n",
            " [3 3]\n",
            " [3 3]\n",
            " [0 0]\n",
            " [9 9]\n",
            " [5 5]\n",
            " [3 9]\n",
            " [2 2]\n",
            " [1 1]\n",
            " [4 4]\n",
            " [2 2]\n",
            " [1 1]\n",
            " [6 6]\n",
            " [8 8]\n",
            " [9 9]\n",
            " [2 2]\n",
            " [4 4]\n",
            " [9 9]\n",
            " [3 3]\n",
            " [7 7]\n",
            " [6 6]\n",
            " [2 2]\n",
            " [3 3]\n",
            " [3 3]\n",
            " [1 1]\n",
            " [6 6]\n",
            " [9 9]\n",
            " [3 3]\n",
            " [6 6]\n",
            " [3 3]\n",
            " [2 2]\n",
            " [2 2]\n",
            " [0 0]\n",
            " [7 7]\n",
            " [6 6]\n",
            " [1 1]\n",
            " [1 1]\n",
            " [9 9]\n",
            " [7 7]\n",
            " [2 2]\n",
            " [7 7]\n",
            " [8 8]\n",
            " [5 5]\n",
            " [5 5]\n",
            " [7 7]\n",
            " [5 5]\n",
            " [2 2]\n",
            " [3 3]\n",
            " [7 7]\n",
            " [2 2]\n",
            " [7 7]\n",
            " [5 5]\n",
            " [5 5]\n",
            " [7 7]\n",
            " [0 0]\n",
            " [9 9]\n",
            " [1 1]\n",
            " [6 6]\n",
            " [5 5]\n",
            " [9 9]\n",
            " [7 7]\n",
            " [4 4]\n",
            " [3 3]\n",
            " [8 8]\n",
            " [0 0]\n",
            " [3 3]\n",
            " [6 6]\n",
            " [4 4]\n",
            " [6 6]\n",
            " [3 3]\n",
            " [2 2]\n",
            " [6 6]\n",
            " [8 8]\n",
            " [8 8]\n",
            " [8 8]\n",
            " [4 4]\n",
            " [6 6]\n",
            " [7 7]\n",
            " [5 5]\n",
            " [2 2]\n",
            " [4 4]\n",
            " [5 5]\n",
            " [3 3]\n",
            " [2 2]\n",
            " [4 4]\n",
            " [6 6]\n",
            " [9 9]\n",
            " [4 4]\n",
            " [5 5]\n",
            " [4 4]\n",
            " [3 3]\n",
            " [4 4]\n",
            " [6 6]\n",
            " [2 2]\n",
            " [9 9]\n",
            " [0 0]\n",
            " [1 1]\n",
            " [7 7]\n",
            " [2 2]\n",
            " [0 0]\n",
            " [9 9]\n",
            " [6 6]\n",
            " [0 0]\n",
            " [4 4]\n",
            " [2 2]\n",
            " [0 0]\n",
            " [7 7]\n",
            " [9 9]\n",
            " [8 8]\n",
            " [5 5]\n",
            " [4 4]\n",
            " [8 8]\n",
            " [2 2]\n",
            " [8 8]\n",
            " [4 4]\n",
            " [3 3]\n",
            " [7 7]\n",
            " [2 2]\n",
            " [6 6]\n",
            " [9 9]\n",
            " [1 1]\n",
            " [5 5]\n",
            " [1 1]\n",
            " [0 0]\n",
            " [8 8]\n",
            " [2 2]\n",
            " [8 1]\n",
            " [9 9]\n",
            " [5 5]\n",
            " [6 6]\n",
            " [2 8]\n",
            " [2 2]\n",
            " [7 7]\n",
            " [2 2]\n",
            " [1 1]\n",
            " [5 5]\n",
            " [1 1]\n",
            " [6 6]\n",
            " [4 4]\n",
            " [5 5]\n",
            " [0 0]\n",
            " [9 9]\n",
            " [4 4]\n",
            " [1 1]\n",
            " [1 1]\n",
            " [7 7]\n",
            " [0 0]\n",
            " [8 8]\n",
            " [9 9]\n",
            " [0 0]\n",
            " [5 5]\n",
            " [4 4]\n",
            " [3 3]\n",
            " [8 8]\n",
            " [8 8]\n",
            " [6 6]\n",
            " [5 5]\n",
            " [3 3]\n",
            " [4 4]\n",
            " [4 4]\n",
            " [4 4]\n",
            " [8 8]\n",
            " [8 8]\n",
            " [7 7]\n",
            " [0 0]\n",
            " [9 9]\n",
            " [6 6]\n",
            " [3 3]\n",
            " [5 5]\n",
            " [2 2]\n",
            " [3 3]\n",
            " [0 0]\n",
            " [8 8]\n",
            " [8 3]\n",
            " [3 3]\n",
            " [1 1]\n",
            " [3 3]\n",
            " [3 3]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [4 4]\n",
            " [6 6]\n",
            " [0 0]\n",
            " [7 7]\n",
            " [7 7]\n",
            " [6 6]\n",
            " [2 2]\n",
            " [0 0]\n",
            " [4 4]\n",
            " [4 4]\n",
            " [2 2]\n",
            " [3 3]\n",
            " [7 7]\n",
            " [1 8]\n",
            " [9 9]\n",
            " [8 8]\n",
            " [6 6]\n",
            " [8 8]\n",
            " [5 5]\n",
            " [6 6]\n",
            " [2 2]\n",
            " [2 2]\n",
            " [3 3]\n",
            " [1 1]\n",
            " [7 7]\n",
            " [7 7]\n",
            " [8 8]\n",
            " [0 0]\n",
            " [3 3]\n",
            " [3 3]\n",
            " [2 2]\n",
            " [1 1]\n",
            " [5 5]\n",
            " [5 5]\n",
            " [9 9]\n",
            " [1 1]\n",
            " [3 3]\n",
            " [7 7]\n",
            " [0 0]\n",
            " [0 0]\n",
            " [7 7]\n",
            " [0 0]\n",
            " [4 4]\n",
            " [5 5]\n",
            " [8 9]\n",
            " [9 3]\n",
            " [3 3]\n",
            " [4 4]\n",
            " [3 3]\n",
            " [1 1]\n",
            " [8 8]\n",
            " [9 9]\n",
            " [8 8]\n",
            " [3 3]\n",
            " [6 6]\n",
            " [2 2]\n",
            " [1 1]\n",
            " [6 6]\n",
            " [2 2]\n",
            " [1 1]\n",
            " [7 7]\n",
            " [5 5]\n",
            " [5 5]\n",
            " [1 1]\n",
            " [9 9]]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "mIEF2XT2ZFyO",
        "outputId": "6368a2ed-a968-4156-845d-6fc7e8780ed1"
      },
      "source": [
        "from sklearn.metrics import accuracy_score\n",
        "print(\"Accuracy of the model: {0}%\".format(accuracy_score(y_test,y_pred)*100))"
      ],
      "execution_count": 20,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Accuracy of the model: 97.11111111111111%\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "PwCRVX3SZuSe",
        "outputId": "ce2d092d-236b-43e9-d67a-cb1dd41983c9"
      },
      "source": [
        "from sklearn import svm\n",
        "model1 = svm.SVC(kernel='linear' )\n",
        "model2 = svm.SVC(kernel='rbf' )\n",
        "model3 = svm.SVC(gamma=0.001)\n",
        "model4 = svm.SVC(gamma=0.001, C=0.1)\n",
        "\n",
        "model1.fit(X_train, y_train)\n",
        "mode12.fit(X_train, y_train)\n",
        "mode13.fit(X_train, y_train)\n",
        "mode14.fit(X_train, y_train)\n",
        "\n",
        "y_predModel1 = model1.predict(X_test)\n",
        "\n",
        "y_predModel3 = mode13.predict(X_test)\n",
        "y_predModel4 = mode14.predict(X_test)\n",
        "\n",
        "print(\"Accuracy of the Model 1: {0}%\".format(accuracy_score(y_test, y_predModel1)*100))\n",
        "#print(\"Accuracy of the Model 2: {0}%\".format(accuracy_score(y_test, y_predMode12)*100))\n",
        "print(\"Accuracy of the Model 3: {}%\".format(accuracy_score(y_test, y_predMode13)*100))\n",
        "print(\"Accuracy of the Model 4: {}%\".format(accuracy_score(y_test, y_predMode14)*100))\n"
      ],
      "execution_count": 32,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Accuracy of the Model 1: 97.11111111111111%\n",
            "Accuracy of the Model 3: 99.55555555555556%\n",
            "Accuracy of the Model 4: 96.66666666666667%\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "kZetzGXpZvcg"
      },
      "source": [
        ""
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}
