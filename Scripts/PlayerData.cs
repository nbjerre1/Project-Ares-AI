using System.Collections.Generic;
using Godot;

[GlobalClass]
public partial class PlayerData : Resource
{
    [Export] public int Gold { get; set; } = 0;
    public ArrowUpgrade? ActiveArrowUpgrade { get; private set; } = null;
  
    public List<ArrowUpgrade> Upgrades { get; } = new();
    public enum ArrowUpgrade
    {
        Fire,
        Ice,
        diagonalShot,
        piercingShot,
        multiShot,
    }

    public void AddGold(int amount)
    {
        Gold += amount;
        GD.Print($"Player gained {amount} Gold. Total Gold: {Gold}");
    }

    public void Save()
    {
        //use user://player_data.tres to save player data in appdata user folder
        ResourceSaver.Save(this, "res://player_data.tres");
    }

    public static PlayerData Load()
    {
        //use user://player_data.tres to save player data in appdata user folder
        GD.Print("Checking for player data at res://player_data.tres");
        if (ResourceLoader.Exists("res://player_data.tres"))
            GD.Print("Loading player data from res://player_data.tres");
        return ResourceLoader.Load<PlayerData>("res://player_data.tres");
       // return new PlayerData();
    }
    public void AddUpgrade(ArrowUpgrade upgrade)
    {
        // Example: Add the upgrade to a list or set a flag
        // Upgrades could be a List<ArrowUpgrade> or similar
        Upgrades.Add(upgrade);
    }
    public bool HasUpgrade(ArrowUpgrade upgrade)
    {
        return Upgrades.Contains(upgrade);
    }

    public void SetActiveArrowUpgrade(ArrowUpgrade upgrade)
    {
        if (HasUpgrade(upgrade))
            ActiveArrowUpgrade = upgrade;
    }
   
    
    public void AddDiagonalShotUpgrade(Player player)
    {
        player.DiagonalShotLevel++;
    }
}
