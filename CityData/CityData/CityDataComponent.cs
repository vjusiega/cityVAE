using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using Rhino.Geometry;

// In order to load the result of this wizard, you will also need to
// add the output bin/ folder of this project to the list of loaded
// folder in Grasshopper.
// You can use the _GrasshopperDeveloperSettings Rhino command for that.

namespace CityData
{
    public class CityDataComponent : GH_Component
    {
        /// <summary>
        /// Each implementation of GH_Component must provide a public 
        /// constructor without any arguments.
        /// Category represents the Tab in which the component will appear, 
        /// Subcategory the panel. If you use non-existing tab or panel names, 
        /// new tabs/panels will automatically be created.
        /// </summary>
        public CityDataComponent()
          : base("CityData", "Nickname",
              "Description",
              "Category", "Subcategory")
        {
        }

        /// <summary>
        /// Registers all the input parameters for this component.
        /// </summary>
        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            pManager.AddTextParameter("filepath", "file", "data filepath", GH_ParamAccess.item);
        }

        /// <summary>
        /// Registers all the output parameters for this component.
        /// </summary>
        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.AddPointParameter("Points", "P", "city points", GH_ParamAccess.list);
            //pManager.AddBrepParameter("Breps", "B", "city breps", GH_ParamAccess.list);
        }

        /// <summary>
        /// This is the method that actually does the work.
        /// </summary>
        /// <param name="DA">The DA object can be used to retrieve data from input parameters and 
        /// to store data in output parameters.</param>
        protected override void SolveInstance(IGH_DataAccess DA)
        {
            int img_size = 100; //size of images we are trying to render
            double conversion = 0.3048 * 1776; //conversion factor of the points. They are currently standardized between 0-1, so multiply by 1776, then by feet to meters factor

            string data = ""; //data is read from file
            string file = ""; //file name is recieved as input
            if (!DA.GetData(0, ref file))
            {
                return;
            }
            FileStream fs = new FileStream(file, FileMode.Open, FileAccess.Read);
            using (StreamReader streamReader = new StreamReader(fs, Encoding.UTF8))
            {
                data = streamReader.ReadToEnd();
            }

            //clean the input
            data = data.Replace("[", " ");
            data = data.Replace("]", " ");
            data = data.Replace("  ", " ");
            data = data.Replace("\r\n", " ");
            data = data.Replace("\n", " ");

            int row_index = 0;
            int col_index = 0;
            int str_start = 0;

            //parse the data into a nested list (to be able to retrieve coordinate information)
            List<List<double>> grid = new List<List<double>>();
            for(int k = 0; k < img_size; k++)
            {
                grid.Add(new List<double>());
            }

            //go through every index of the data
            for(int i = 0; i < data.Length; i++)
            {
                //if we are getting a subset of the data, then we stop parsing it once we get the desired size
                if(row_index == img_size)
                {
                    break;
                }

                //values are seperated by spaces
                //when space is found, convert substring to double and place in list
                if (data.Substring(i, 1) == " ")
                {
                    double height = Convert.ToDouble(data.Substring(str_start, i - str_start));
                    if (height == -9999.0) //specific to how our data was generated
                    {
                        height = 0;
                    }
                    grid[row_index].Add(height);

                    str_start = i + 1;
                    col_index = col_index + 1; 

                    if(col_index == img_size)
                    {
                        row_index += 1;
                        col_index = 0;
                    }
                }
            }

            List<GH_Point> points = new List<GH_Point>();
            for (int row = 0; row < grid.Count; row++)
            {
                for (int col = 0; col < grid[0].Count; col++)
                {
                    //the value is zero we create one point for it at the center of its grid location
                    if (grid[row][col] == 0)
                    {
                        Point3d p1 = new Point3d(col * 2 + 1, row * 2 + 1, grid[row][col] * conversion);
                        GH_Point gh_p1 = new GH_Point(p1);
                        points.Add(gh_p1);
                    }
                    //for other values we generated four points for the corners of the grid point that this value occupies
                    //for the purpose of generating clearer/sharper renderings 
                    else
                    {
                        Point3d p1 = new Point3d(col * 2 + 1, row * 2 + 1, grid[row][col] * conversion);
                        GH_Point gh_p1 = new GH_Point(p1);
                        points.Add(gh_p1);

                        Point3d p2 = new Point3d((col * 2) + 2, (row * 2) + 2, grid[row][col] * conversion);
                        GH_Point gh_p2 = new GH_Point(p2);
                        points.Add(gh_p2);

                        Point3d p3 = new Point3d(col * 2, (row * 2) + 2, grid[row][col] * conversion);
                        GH_Point gh_p3 = new GH_Point(p3);
                        points.Add(gh_p3);

                        Point3d p4 = new Point3d((col * 2) + 2, row * 2, grid[row][col] * conversion);
                        GH_Point gh_p4 = new GH_Point(p4);
                        points.Add(gh_p4);
                    }
                }
            }
            DA.SetDataList(0, points);
        }



        /// <summary>
        /// Provides an Icon for every component that will be visible in the User Interface.
        /// Icons need to be 24x24 pixels.
        /// </summary>
        protected override System.Drawing.Bitmap Icon
        {
            get
            {
                // You can add image files to your project resources and access them like this:
                //return Resources.IconForThisComponent;
                return null;
            }
        }

        /// <summary>
        /// Each component must have a unique Guid to identify it. 
        /// It is vital this Guid doesn't change otherwise old ghx files 
        /// that use the old ID will partially fail during loading.
        /// </summary>
        public override Guid ComponentGuid
        {
            get { return new Guid("3dc4f9ed-c688-4334-b12d-8abef1c54eb5"); }
        }
    }
}
