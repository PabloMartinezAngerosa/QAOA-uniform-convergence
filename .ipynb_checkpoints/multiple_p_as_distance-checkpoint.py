{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "f25878c9",
   "metadata": {},
   "outputs": [],
   "source": [
    "from tsp_qaoa import test_solution\n",
    "from qiskit.visualization import plot_histogram\n",
    "import networkx as nx\n",
    "import numpy as np\n",
    "import json\n",
    "import csv\n",
    "# Array of JSON Objects"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "28d13279",
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Vuelve a llamar a test_solution\n",
      "0.537411136357747\n",
      "[0, 1]\n",
      "[(0, 1)]\n",
      "[0, 1]\n",
      "     fun: 10.39246406840091\n",
      "   maxcv: 0.0\n",
      " message: 'Optimization terminated successfully.'\n",
      "    nfev: 54\n",
      "  status: 1\n",
      " success: True\n",
      "       x: array([2.01735703, 5.73124225])\n",
      "*************************\n",
      "En el algoritmo 2 (Marina) el valor esperado como resultado es 10.508445063673703\n",
      "El valor minimo de todos los evaluados es 1.074822272715494 se evaluaron un total de 16\n",
      "El vector minimo es 1001\n",
      "media_minima\n"
     ]
    },
    {
     "ename": "KeyError",
     "evalue": "0",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mKeyError\u001b[0m                                  Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-12-d4340f025863>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m     31\u001b[0m             \u001b[0mUNIFORM_CONVERGENCE_P\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mappend\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mconvergence_min\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     32\u001b[0m             \u001b[0mprint\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"media_minima\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 33\u001b[0;31m             \u001b[0mprint\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mconvergence_min\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m\"mean\"\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     34\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     35\u001b[0m         \u001b[0mcauchy_function_nk\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mUNIFORM_CONVERGENCE_P\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mlen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mUNIFORM_CONVERGENCE_P\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m-\u001b[0m \u001b[0;36m1\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mKeyError\u001b[0m: 0"
     ]
    }
   ],
   "source": [
    "header = ['instance','p','distance', 'mean']\n",
    "length_p = 3\n",
    "length_instances = 2\n",
    "\n",
    "\n",
    "with open('qaoa_multiple_p_distance.csv', 'w', encoding='UTF8') as f:\n",
    "    writer = csv.writer(f)\n",
    "\n",
    "    # write the header\n",
    "    writer.writerow(header)\n",
    "    \n",
    "    \n",
    "    instance_index = 0\n",
    "    for instance in range(length_instances):\n",
    "        instance_index += 1\n",
    "        first_p = False\n",
    "        UNIFORM_CONVERGENCE_P = []\n",
    "        UNIFORM_CONVERGENCE_SAMPLE = []\n",
    "        for p in range(length_p):\n",
    "            p = p+1\n",
    "            if first_p == False:\n",
    "                print(\"Vuelve a llamar a test_solution\")\n",
    "                job_2, G, UNIFORM_CONVERGENCE_SAMPLE = test_solution(p=p)\n",
    "                first_p = True\n",
    "            else:\n",
    "                job_2, G, UNIFORM_CONVERGENCE_SAMPLE = test_solution(grafo=G, p=p)\n",
    "\n",
    "            # Sort the JSON data based on the value of the brand key\n",
    "            UNIFORM_CONVERGENCE_SAMPLE.sort(key=lambda x: x[\"mean\"])\n",
    "            convergence_min = UNIFORM_CONVERGENCE_SAMPLE[0]\n",
    "            UNIFORM_CONVERGENCE_P.append(convergence_min)\n",
    "            print(\"media_minima\")\n",
    "            print(convergence_min[\"mean\"])\n",
    "        \n",
    "        cauchy_function_nk = UNIFORM_CONVERGENCE_P[len(UNIFORM_CONVERGENCE_P) - 1]\n",
    "        p_index = 0\n",
    "        for p_state in UNIFORM_CONVERGENCE_P:\n",
    "            p_index += 1\n",
    "            print(p_index)\n",
    "            mean = p_state[\"mean\"]\n",
    "            print(p_state)\n",
    "            print(mean)\n",
    "            distance_p_cauchy_function_nk = np.max(np.abs(cauchy_function_nk[\"probabilities\"] - p_state[\"probabilities\"]))\n",
    "            writer.writerow([instance_index, p_index, distance_p_cauchy_function_nk, mean])\n",
    "        \n",
    "    \n",
    "    \n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "add0ec1a",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
