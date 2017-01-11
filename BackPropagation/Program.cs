using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace BackPropagation
{
    class Program
    {
        static void Main(string[] args)
        {
            Perceptron percaptron = new Perceptron();
            percaptron.Train();
            Console.ReadLine();
        }
    }
}
