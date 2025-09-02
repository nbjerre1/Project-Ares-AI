using Godot;
using System;

public partial class StopwatchLabel : Label
{
    private double timeElapsed = 0.0;
    private const double MaxTime = 1800.0; // 30 minutes in seconds
    public static double GameTime { get; private set; }

    public override void _Process(double delta)
    {
        // Count up until 30 minutes
        if (timeElapsed < MaxTime)
        {
            timeElapsed += delta;
        }
        else
        {
            timeElapsed = MaxTime; // Clamp at 30 minutes
        }
        GameTime = timeElapsed; // Update static property
        Text = FormatSeconds(timeElapsed, true); // set to false if you don’t want milliseconds
    }

    private string FormatSeconds(double time, bool useMilliseconds)
    {
        int minutes = (int)(time / 60);
        int seconds = (int)(time % 60);

        if (!useMilliseconds)
            return $"{minutes:D2}:{seconds:D2}";

        int milliseconds = (int)((time % 1) * 100);
        return $"{minutes:D2}:{seconds:D2}:{milliseconds:D2}";
    }
}