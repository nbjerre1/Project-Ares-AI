using Godot;
using System;

public partial class Gold : Area2D
{
    [Export] public int GoldAmount { get; set; } = 1; // Amount of gold this coin gives

    public override void _Ready()
    {
        BodyEntered += OnBodyEntered;
    }

    private void OnBodyEntered(Node2D body)
    {
        if (body is Player player)
        {
            GD.Print("Gold picked up!");
            player.AddGold(GoldAmount);
            QueueFree();
        }
    }

}
