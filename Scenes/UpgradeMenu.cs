using Godot;
using System;

public partial class UpgradeMenu : Control
{
    public override void _Ready()
    {
        ProcessMode = ProcessModeEnum.Always;
        GetNode<Button>("CanvasLayer/Button").Pressed += OnFireArrowUpgrade;
        GetNode<Button>("CanvasLayer/Button2").Pressed += OnIceArrowUpgrade;
        GetNode<Button>("CanvasLayer/Button3").Pressed += OnDiagonalShotUpgrade; // Add this line
    }

    private void OnFireArrowUpgrade()
    {
        GD.Print("Fire Arrow Upgrade Selected");
        var player = GetNode<Player>("/root/Main/Player");
        player.Data.AddUpgrade(PlayerData.ArrowUpgrade.Fire);
        player.Data.SetActiveArrowUpgrade(PlayerData.ArrowUpgrade.Fire); // Set as active
        CloseMenu();
    }

    private void OnIceArrowUpgrade()
    {
        var player = GetNode<Player>("/root/Main/Player");
        player.Data.AddUpgrade(PlayerData.ArrowUpgrade.Ice);
        player.Data.SetActiveArrowUpgrade(PlayerData.ArrowUpgrade.Ice); // Set as active
        CloseMenu();
    }
    private void OnDiagonalShotUpgrade()
    {
        var player = GetNode<Player>("/root/Main/Player");
        player.Data.AddDiagonalShotUpgrade(player);
        CloseMenu();
    }

    private void CloseMenu()
    {
        QueueFree(); // Removes the menu from the scene tree
        GetTree().Paused = false;
    }
}
